#define GLM_SWIZZLE
#include <sstream>
#include <cuda_runtime.h>
#include "lfLoader.h"
#include "interpolator.h"
#include "kernels.cu"
#include "libs/loadingBar/loadingbar.hpp"

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
}

int Interpolator::createTextureObject(const uint8_t *data, glm::ivec3 size)
{
    constexpr size_t CHANNELS{4};
    cudaChannelFormatDesc channels = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray *arr;
    cudaMallocArray(&arr, &channels, size.x, size.y, 0);
    cudaMemcpy2DToArray(arr, 0, 0, data, size.x*CHANNELS*sizeof(uint8_t), size.x, size.y, cudaMemcpyHostToDevice);

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = arr;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;
    cudaTextureObject_t texObj{0};
    cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL);
    return texObj;
}

std::pair<int, int*> Interpolator::createSurfaceObject(glm::ivec3 size)
{
    cudaChannelFormatDesc channels = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned); 
    cudaArray *arr;
    cudaMallocArray(&arr, &channels, size.x, size.y, cudaArraySurfaceLoadStore);

    cudaResourceDesc surfRes;
    surfRes.res.array.array = arr;
    cudaSurfaceObject_t surfObj = 0;
    cudaCreateSurfaceObject(&surfObj, &surfRes);
    return {surfObj, reinterpret_cast<int*>(arr)};
}

void Interpolator::loadGPUData()
{
    LfLoader lfLoader;
    colsRows = lfLoader.getColsRows();
    lfLoader.loadData(input);
    glm::ivec3 resolution = lfLoader.imageResolution();

    std::cout << "Uploading data to GPU...";
    LoadingBar bar(lfLoader.imageCount()+1);

    for(int i=0; i<viewCount; i++)
    {
        auto surface = createSurfaceObject(resolution);
        surfaces.push_back(surface.first);  
        outputArrays.push_back(surface.second);
    }
    bar.add();
 
    for(int col=0; col<colsRows.x; col++)
        for(int row=0; row<colsRows.y; row++)
        {
            textures.push_back(createTextureObject(lfLoader.image({col, row}).data(), resolution)); 
            bar.add();
        }
}

void Interpolator::loadGPUConstants(glm::ivec2 imgResolution, glm::ivec2 gridSize)
{
    std::vector<int> values{imgResolution.x, imgResolution.y, gridSize.x, gridSize.y};
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

void Interpolator::loadGPUWeights(glm::vec4 startEndPoints)
{
    cudaMalloc(reinterpret_cast<void **>(&weights), sizeof(half)*viewCount*colsRows.x*colsRows.y);
    auto trajectory = generateTrajectory(startEndPoints);
    std::vector<half> weightsMatrix;
    for(auto const &coord : trajectory)
    {
        auto floatWeightsLine = generateWeights(coord);
        std::vector<half> weightsLine;
        for(const auto & w : floatWeightsLine)
            weightsLine.push_back(static_cast<half>(w));
        weightsMatrix.insert(weightsMatrix.end(), weightsLine.begin(), weightsLine.end());
    }
    cudaMemcpy(weights, weightsMatrix.data(), weightsMatrix.size(), cudaMemcpyHostToDevice);
}
 
void Interpolator::interpolate(std::string outputPath, std::string trajectory, bool tensor)
{
    auto trajectoryPoints = interpretTrajectory(trajectory);
    loadGPUWeights(trajectoryPoints);
}

void interpolateTensor(std::string outputPath, std::string trajectory)
{

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
        absolute[i] = value*colsRows[i%2];
        i++;
    }
    return absolute;
}
