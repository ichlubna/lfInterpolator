#include <algorithm>
#include <stdexcept>
#define GLM_FORCE_SWIZZLE
#include <sstream>
#include <cuda_runtime.h>
#include "interpolator.h"
#include "kernels.cu"
#include "lfLoader.h"
#include "libs/loadingBar/loadingbar.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/stb_image_write.h"

class Timer
{
    public:
    Timer()
    {    
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent);
    }
    float stop()
    {
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        float time = 0;
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return time; 
    };
    private:
    cudaEvent_t startEvent, stopEvent;
};

Interpolator::Interpolator(std::string inputPath) : input{inputPath}
{
    init();
}

Interpolator::~Interpolator()
{
    cudaDeviceReset();
}

void Interpolator::init()
{
    viewCount = Kernels::VIEW_COUNT;
    loadGPUData();
    sharedSize = sizeof(half)*colsRows.x*colsRows.y*viewCount;
}

int Interpolator::createTextureObject(const uint8_t *data, glm::ivec3 size)
{
    cudaChannelFormatDesc channels = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray *arr;
    cudaMallocArray(&arr, &channels, size.x, size.y);
    cudaMemcpy2DToArray(arr, 0, 0, data, size.x*size.z, size.x*size.z, size.y, cudaMemcpyHostToDevice);
    
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = arr;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
    cudaTextureObject_t texObj{0};
    cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL);
    return texObj;
}

std::pair<int, int*> Interpolator::createSurfaceObject(glm::ivec3 size, const uint8_t *data)
{
    auto arr = loadImageToArray(data, size);
    cudaResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = reinterpret_cast<cudaArray*>(arr);
    cudaSurfaceObject_t surfObj = 0;
    cudaCreateSurfaceObject(&surfObj, &surfRes);
    return {surfObj, arr};
}

int* Interpolator::loadImageToArray(const uint8_t *data, glm::ivec3 size)
{
    cudaChannelFormatDesc channels = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned); 
    cudaArray *arr;
    cudaMallocArray(&arr, &channels, size.x, size.y, cudaArraySurfaceLoadStore);
    if(data != nullptr)
        cudaMemcpy2DToArray(arr, 0, 0, data, size.x*size.z, size.x*size.z, size.y, cudaMemcpyHostToDevice);
    return reinterpret_cast<int*>(arr);
}

void Interpolator::loadGPUData()
{
    LfLoader lfLoader;
    lfLoader.loadData(input);
    colsRows = lfLoader.getColsRows();
    resolution = lfLoader.imageResolution();

    std::cout << "Uploading data to GPU..." << std::endl;
    LoadingBar bar(lfLoader.imageCount()+viewCount+Kernels::MAP_COUNT);
    
    std::vector<cudaSurfaceObject_t> surfaces;
    for(int col=0; col<colsRows.x; col++)
        for(int row=0; row<colsRows.y; row++)
        {
            auto surface = createSurfaceObject(resolution, lfLoader.image({col, row}).data());
            surfaces.push_back(surface.first);  
            surfaceInputArrays.push_back(surface.second);
            bar.add();
        }

    for(int i=0; i<viewCount+Kernels::MAP_COUNT; i++)
    {
        auto surface = createSurfaceObject(resolution);
        surfaces.push_back(surface.first);  
        surfaceOutputArrays.push_back(surface.second);
        bar.add();
    }
    int inputOffset{colsRows.x*colsRows.y};
    cudaMemcpyToSymbol(Kernels::inputSurfaces, surfaces.data(), sizeof(cudaSurfaceObject_t)*inputOffset);
    cudaMemcpyToSymbol(Kernels::outputSurfaces, surfaces.data()+inputOffset, sizeof(cudaSurfaceObject_t)*viewCount);
    cudaMemcpyToSymbol(Kernels::mapSurfaces, surfaces.data()+inputOffset+viewCount, sizeof(cudaSurfaceObject_t)*Kernels::MAP_COUNT);
}

void Interpolator::loadGPUConstants()
{
    constexpr int PIXEL_SIZE_FACTOR{100};
    int2 blockRadius {resolution.x/PIXEL_SIZE_FACTOR, resolution.y/PIXEL_SIZE_FACTOR};
    if((blockRadius.x % 2) != 0)
        blockRadius.x++;
    if((blockRadius.y % 2) != 0)
        blockRadius.y++;
    std::vector<int> values{resolution.x, resolution.y,
                            colsRows.x, colsRows.y, viewCount,
                            colsRows.x*colsRows.y, colsRows.x*colsRows.y*viewCount,
                            focus, range, blockRadius.x, blockRadius.y};
    cudaMemcpyToSymbol(Kernels::constants, values.data(), values.size() * sizeof(int));
}

std::vector<float> Interpolator::generateWeights(glm::vec2 coords)
{
    auto maxDistance = glm::distance(glm::vec2(0,0), glm::vec2(colsRows));
    float weightSum{0};
    std::vector<float> weightVals;
    for(int col=0; col<colsRows.x; col++)
        for(int row=0; row<colsRows.y; row++)
        {
            float weight = maxDistance - glm::distance(coords, glm::vec2(col, row));
            weightSum += weight;
            weightVals.push_back(weight);
        }
    for(auto &weight : weightVals)
        weight /= weightSum; 
    return weightVals;
}

std::vector<glm::vec2> Interpolator::generateTrajectory(glm::vec4 startEndPoints)
{
    glm::vec2 step = (startEndPoints.zw() - startEndPoints.xy())/static_cast<float>(viewCount);
    std::vector<glm::vec2> trajectory;
    for(int i=0; i<viewCount; i++)
        trajectory.push_back(startEndPoints.xy()+step*static_cast<float>(i));
    return trajectory;
}

bool compareDistances(std::pair<float, int> a, std::pair<float, int> b)
{
    return (a.first < b.first);
}

void Interpolator::selectFocusMapViews(glm::vec4 startEndPoints)
{
    std::vector<std::pair<float, int>> distances;
    glm::vec2 center = startEndPoints.xy()+(startEndPoints.zw() - startEndPoints.xy())*0.5f;
    for(int col=0; col<colsRows.x; col++)
        for(int row=0; row<colsRows.y; row++)
            distances.push_back({glm::distance({col, row}, center), distances.size()});
        
    std::sort(distances.begin(), distances.end(), compareDistances);
    std::vector<int> ids;
    for(int i=0; i<Kernels::FOCUS_MAP_IDS_COUNT; i++)
        ids.push_back(distances[i].second);
    cudaMemcpyToSymbol(Kernels::focusMapIDs, ids.data(), ids.size() * sizeof(int));
}

void Interpolator::loadGPUWeights(glm::vec4 startEndPoints)
{
    cudaMalloc(reinterpret_cast<void **>(&weights), sizeof(half)*viewCount*colsRows.x*colsRows.y);
    auto trajectory = generateTrajectory(startEndPoints);
    std::vector<half> weightsMatrix;
    for(auto const &view : trajectory)
    {
        auto floatWeightsLine = generateWeights(view);
        std::vector<half> weightsLine;
        for(const auto & w : floatWeightsLine)
            weightsLine.push_back(static_cast<half>(w));
        
        weightsMatrix.insert(weightsMatrix.end(), weightsLine.begin(), weightsLine.end());
    }
    cudaMemcpy(weights, weightsMatrix.data(), weightsMatrix.size()*sizeof(half), cudaMemcpyHostToDevice);
}

void Interpolator::loadGPUOffsets()
{
    std::vector<int2> focusedOffsets;
    std::vector<float2> offsets;
    glm::vec2 maxOffset{colsRows-glm::ivec2(1)};
    glm::vec2 center{maxOffset*glm::vec2(0.5)}; 
    for(int col=0; col<colsRows.x; col++)
        for(int row=0; row<colsRows.y; row++)
        {
            glm::vec2 position{col, row};
            glm::vec2 offset{(center-position)/maxOffset};
            offsets.push_back({offset.x, offset.y});
            glm::ivec2 rounded = glm::round(offset*static_cast<float>(focus));
            focusedOffsets.push_back({rounded.x, rounded.y});
        }
    cudaMemcpyToSymbol(Kernels::focusedOffsets, focusedOffsets.data(), focusedOffsets.size() * sizeof(int2));
    cudaMemcpyToSymbol(Kernels::offsets, offsets.data(), offsets.size() * sizeof(float2));
}

void Interpolator::interpolate(std::string outputPath, std::string trajectory, float inFocus, float inRange, std::string method)
{
    focus = inFocus;
    range = inRange;
    loadGPUOffsets();
    auto trajectoryPoints = interpretTrajectory(trajectory);
    loadGPUWeights(trajectoryPoints);
    selectFocusMapViews(trajectoryPoints);
    loadGPUConstants();
    
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(resolution.x/dimBlock.x+1, resolution.y/dimBlock.y+1, 1);

    if(inRange > 0)
        Kernels::estimateFocusMap<<<dimGrid, dimBlock, sharedSize>>>();
    
    std::cout << "Elapsed time: "<< std::endl;
    float avgTime{0};
    for(size_t i=0; i<kernelBenchmarkRuns; i++)
    {
        Timer timer;
        if(method == "TEN_WM")
        {
            size_t tensorSharedSize = sharedSize+(32*16+32*8)*sizeof(half)*(dimBlock.x*dimBlock.y/32)*3;
            if(range > 0)
                Kernels::processTensor<true><<<dimGrid, dimBlock, tensorSharedSize>>>(reinterpret_cast<half*>(weights));
            else
                Kernels::processTensor<false><<<dimGrid, dimBlock, tensorSharedSize>>>(reinterpret_cast<half*>(weights));
        }
        else if(method == "TEN_OP")
        {
            //Kernels::processTensor<<<dimGrid, dimBlock, sharedSize>>>(reinterpret_cast<half*>(weights));
        }
        else if(method == "STD")
        {
            if(range > 0)
                Kernels::process<true><<<dimGrid, dimBlock, sharedSize>>>(reinterpret_cast<half*>(weights));
            else
                Kernels::process<false><<<dimGrid, dimBlock, sharedSize>>>(reinterpret_cast<half*>(weights));
        }
        else
            throw std::runtime_error("The specified interpolation method does not exist!");
        avgTime += timer.stop();
    }
    std::cout << "Average time of " << std::to_string(kernelBenchmarkRuns) << " runs: " << avgTime/kernelBenchmarkRuns  << " ms" << std::endl;
    storeResults(outputPath);
}

void Interpolator::storeResults(std::string path)
{
    std::cout << "Storing results..." << std::endl;
    int count = viewCount;
    if(range > 0)
        count += 1;
    LoadingBar bar(count);
    std::vector<uint8_t> data(resolution.x*resolution.y*resolution.z, 255);
    for(int i=0; i<count; i++) 
    {
        cudaMemcpy2DFromArray(data.data(), resolution.x*resolution.z, reinterpret_cast<cudaArray*>(surfaceOutputArrays[i]), 0, 0, resolution.x*resolution.z, resolution.y, cudaMemcpyDeviceToHost);
        auto fileName = std::filesystem::path(path)/(std::to_string(i)+".png");
        if(i >= viewCount)
            fileName = std::filesystem::path(path)/("map"+std::to_string(i-viewCount)+".png");
        stbi_write_png(fileName.c_str(), resolution.x, resolution.y, resolution.z, data.data(), resolution.x*resolution.z);
        bar.add();
    }
}

glm::vec4 Interpolator::interpretTrajectory(std::string trajectory)
{
    constexpr char delim{','};
    std::vector <std::string> numbers;
    std::stringstream a(trajectory); 
    std::string b; 
    while(getline(a, b, delim))
    {
        numbers.push_back(b);
    }
    glm::vec4 absolute;    
    int i{0};
    for (const auto &number : numbers)
    {
        float value = std::stof(number);
        absolute[i] = value*(colsRows[i%2]-1);
        i++;
    }
    return absolute;
}

