#include <glm/glm.hpp>
#include <cuda_fp16.h>
#include <mma.h>
//#include "libs/CudaTensorLibrary/tensor.cu"

namespace Kernels
{
__device__ constexpr bool GUESS_HANDLES{false};

__device__ constexpr int CHANNELS{3};
__device__ constexpr int CONSTANTS_COUNT{11};
__device__ constexpr int VIEW_COUNT{8};
__device__ constexpr int VIEW_PORTIONS{2};
__device__ constexpr int VIEW_TOTAL_COUNT{VIEW_PORTIONS * VIEW_COUNT};
__constant__ int constants[CONSTANTS_COUNT];
__constant__ float inFocus;
__constant__ float inRange;

__device__ int2 imgRes()
{
    return {constants[0], constants[1]};
}
__device__ int2 colsRows()
{
    return{constants[2], constants[3]};
}
__device__ int2 weightsRes()
{
    return {constants[5], VIEW_TOTAL_COUNT};
}
__device__ int weightsSize()
{
    return constants[6];
}
__device__ int gridSize()
{
    return constants[5];
}
__device__ int focus()
{
    return constants[7];
}
__device__ int focusRange()
{
    return constants[8];
}
__device__ int2 blockRadius()
{
    return {constants[9], constants[10]};
}
__device__ float normalizedFocus()
{
    return inFocus;
}
__device__ float normalizedRange()
{
    return inRange;
}

__device__ constexpr int MAX_IMAGES{256};
__device__ constexpr int MAX_SURFACES{256};
__device__ constexpr int MAP_COUNT{2};
__constant__ int2 focusedOffsets[MAX_IMAGES];
__constant__ float2 offsets[MAX_IMAGES];
__constant__ cudaSurfaceObject_t inputSurfaces[MAX_SURFACES];
__constant__ cudaSurfaceObject_t outputSurfaces[VIEW_TOTAL_COUNT];
__constant__ cudaSurfaceObject_t mapSurfaces[MAP_COUNT];
__device__ constexpr int FOCUS_MAP_IDS_COUNT{32};
__constant__ int focusMapIDs[FOCUS_MAP_IDS_COUNT];
extern __shared__ half localMemory[];

__device__ int2 focusCoords(int2 coords, int imageID)
{
    auto offset = focusedOffsets[imageID];
    return {coords.x + offset.x, coords.y + offset.y};
}

__device__ int2 focusCoords(int2 coords, int imageID, float focus)
{
    auto offset = offsets[imageID];
    return {static_cast<int>(coords.x + focus * offset.x), static_cast<int>(coords.y + focus * offset.y)};
}

__device__ int linearCoords(int2 coords, int width)
{
    return coords.y * width + coords.x;
}

template <typename T>
__device__ static void loadWeightsSync(T *inData, T *data)
{
    int threadsCount = blockDim.x * blockDim.y;
    int batchSize = weightsSize() / threadsCount / 2;
    int id = batchSize * linearCoords(int2{static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y)}, blockDim.x);
    if(id < weightsSize() / 2)
        for(int batch = 0; batch < batchSize; batch++)
        {
            int *intLocal = reinterpret_cast<int *>(data);
            int *intIn = reinterpret_cast<int *>(inData);
            intLocal[id + batch] = intIn[id + batch];
        }
    __syncthreads();
}

__device__ bool coordsOutside(int2 coords)
{
    int2 resolution = imgRes();
    return (coords.x >= resolution.x || coords.y >= resolution.y);
}

__device__ int2 getImgCoords()
{
    int2 coords;
    coords.x = (threadIdx.x + blockIdx.x * blockDim.x);
    coords.y = (threadIdx.y + blockIdx.y * blockDim.y);
    return coords;
}

__device__ uchar4 loadPx(int imageID, int2 coords)
{
    constexpr int MULT_FOUR_SHIFT{2};
    if constexpr(GUESS_HANDLES)
        return surf2Dread<uchar4>(imageID + 1, coords.x << MULT_FOUR_SHIFT, coords.y, cudaBoundaryModeClamp);
    else
        return surf2Dread<uchar4>(inputSurfaces[imageID], coords.x << MULT_FOUR_SHIFT, coords.y, cudaBoundaryModeClamp);
}

__device__ unsigned char loadPxFromMap(int mapID, int2 coords)
{
    constexpr int MULT_FOUR_SHIFT{2};
    return surf2Dread<uchar4>(mapSurfaces[mapID], coords.x << MULT_FOUR_SHIFT, coords.y, cudaBoundaryModeClamp).x;
}

__device__ float loadFocusFromMap(int mapID, int2 coords)
{
    return normalizedFocus() + (static_cast<float>(loadPxFromMap(mapID, coords)) / UCHAR_MAX) * normalizedRange();
}

/*
__constant__ cudaTextureObject_t inputTextures[MAX_SURFACES];
__device__ uchar4 loadPx(int imageID, int2 inCoords)
{
    float2 coords{static_cast<float>(inCoords.x), static_cast<float>(inCoords.y)};
    if constexpr (GUESS_HANDLES)
        return tex2D<uchar4>(imageID+1, coords.x+0.5f, coords.y+0.5f);
    else
        return tex2D<uchar4>(inputTextures[imageID], coords.x, coords.y);
}
*/

__device__ void storePx(uchar4 px, int imageID, int2 coords)
{
    if constexpr(GUESS_HANDLES)
        surf2Dwrite<uchar4>(px, imageID + 1 + gridSize(), coords.x * sizeof(uchar4), coords.y);
    else
        surf2Dwrite<uchar4>(px, outputSurfaces[imageID], coords.x * sizeof(uchar4), coords.y);
}

__device__ void storePxToMap(uchar4 px, int mapID, int2 coords)
{
    surf2Dwrite<uchar4>(px, mapSurfaces[mapID], coords.x * sizeof(uchar4), coords.y);
}

namespace FocusMap
{

__device__ float distance(float *a, float *b)
{
    return fmaxf(fmaxf(fabsf(a[0] - b[0]), fabsf(a[1] - b[1])), fabsf(a[2] - b[2]));
}
/*
    template<typename T>
 class ElementRange
        {
            private:
            float n{0};
            PixelArray<T> m;
            float m2{0};

            public:
            __device__ void add(PixelArray<T> val)
            {
               float dist = distance(m, val);
               n++;
               PixelArray delta = val-m;
               m += delta/static_cast<float>(n);
               //m2 += distance * Pixel::distance(m, val);
               m2 = __fmaf_rn(dist, distance(m, val), m2);

            }
            __device__ float dispersionAmount()
            {
                return m2/(n-1);
            }
            __device__ ElementRange& operator+=(const PixelArray<T>& rhs){

              add(rhs);
              return *this;
            }
        };
*/
template<typename T>
class ElementRange
{
    private:
        float minCol[CHANNELS] {FLT_MAX, FLT_MAX, FLT_MAX};
        float maxCol[CHANNELS] {FLT_MIN, FLT_MIN, FLT_MIN};

    public:
        __device__ void add(uchar4 val)
        {
            auto valArr = reinterpret_cast<unsigned char *>(&val);
            for(int channel = 0; channel < CHANNELS; channel++)
            {
                minCol[channel] = fminf(minCol[channel], valArr[channel]);
                maxCol[channel] = fmaxf(maxCol[channel], valArr[channel]);
            }
        }
        __device__ float dispersionAmount()
        {
            return distance(minCol, maxCol);
        }
};

__device__ float focusDispersion(float focus, int2 coords)
{
    int2 radius = blockRadius();
    constexpr int BLOCK_DIAMETER{3};
    constexpr int BLOCK_SIZE{BLOCK_DIAMETER * BLOCK_DIAMETER};
    ElementRange<float> dispersions[BLOCK_SIZE];

    for(int viewID = 0; viewID < FOCUS_MAP_IDS_COUNT; viewID++)
    {
        int gridID{focusMapIDs[viewID]};
        int i{0};
        int2 focusedCoords = focusCoords(coords, gridID, focus);
        for(int x = focusedCoords.x - radius.x; x <= focusedCoords.x + radius.x; x += radius.x)
            for(int y = focusedCoords.y - radius.y; y <= focusedCoords.y + radius.y; y += radius.y)
                dispersions[i++].add(loadPx(gridID, {x, y}));
    }

    float finalDispersion{0};
    for(int i = 0; i < BLOCK_SIZE; i++)
        finalDispersion += dispersions[i].dispersionAmount();
    return finalDispersion;
}

class MinDispersion
{
    private:
        float dispersion{FLT_MAX};
        float focus{0};
    public:
        __device__ void add(float newFocus, float newDispersion)
        {
            if(newDispersion < dispersion)
            {
                focus = newFocus;
                dispersion = newDispersion;
            }
        }
        __device__ float getBestFocus()
        {
            return focus;
        }
};

__global__ void estimate()
{
    int2 coords = getImgCoords();
    if(coordsOutside(coords))
        return;

    constexpr int STEPS{32};
    float step = normalizedRange() / (STEPS-1);
    MinDispersion minimum;
    for(int i=0; i < STEPS; i++)
    {
        float f = normalizedFocus()+step*i;
        minimum.add(f, focusDispersion(f, coords));
    }

    float bestFocus = minimum.getBestFocus();
    float normalizedMapFocus = (bestFocus - normalizedFocus()) / normalizedRange();
    unsigned char mapFocus{static_cast<unsigned char>(round(normalizedMapFocus * UCHAR_MAX))};
    storePxToMap({mapFocus, mapFocus, mapFocus, UCHAR_MAX}, 0, coords);
    //loadPxFromMap(0, coords);
}
}

namespace Standard
{

__device__ void addWeighted(float3 *base, half weight, uchar4 value)
{
    auto baseArr = reinterpret_cast<float *>(base);
    auto valueArr = reinterpret_cast<unsigned char *>(&value);
    for(int j = 0; j < CHANNELS; j++)
        //baseArr[j] += static_cast<float>(valueArr[j])*static_cast<float>(weight);
        baseArr[j] = __fmaf_rn(static_cast<float>(valueArr[j]), weight, baseArr[j]);
}

__device__ uchar4 uch4(float3 *value)
{
    float *valueArr = reinterpret_cast<float *>(value);
    uchar4 result;
    auto data = reinterpret_cast<unsigned char *>(&result);
    for(int i = 0; i < CHANNELS; i++)
        data[i] = __float2int_rn(valueArr[i]);
    result.w = UCHAR_MAX;
    return result;
}

template<bool allFocus>
__global__ void process(half *weights)
{
    auto localWeights = localMemory;
    loadWeightsSync<half>(weights, localWeights);

    int2 coords = getImgCoords();
    if(coordsOutside(coords))
        return;

    float3 sum[VIEW_TOTAL_COUNT] = {{0, 0, 0}};
    int2 focusedCoords;
    float focusValue{};
    if constexpr(allFocus)
        focusValue = loadFocusFromMap(0, coords);

    for(int gridID = 0; gridID < gridSize(); gridID++)
    {
        if constexpr(allFocus)
            focusedCoords = focusCoords(coords, gridID, focusValue);
        else
            focusedCoords = focusCoords(coords, gridID);

        auto px{loadPx(gridID, focusedCoords)};
        for(int viewID = 0; viewID < VIEW_TOTAL_COUNT; viewID++)
            addWeighted(&sum[viewID], localWeights[linearCoords({gridID, viewID}, weightsRes().x)], px);
    }

    for(int viewID = 0; viewID < VIEW_TOTAL_COUNT; viewID++)
        storePx(uch4(&sum[viewID]), viewID, coords);
}
}

namespace Tensors
{

constexpr int WARP_WIDTH{32};
constexpr int WARP_COUNT = 256 / WARP_WIDTH;
constexpr int PIXELS{32}, VIEWS{8}, IMAGES{16};
constexpr int PIXEL_MATRIX_SIZE{PIXELS * IMAGES};

template<bool allFocus>
__device__ void loadPixels(int batch, int2 coords, unsigned char *destinationPixels, float focusValue)
{
    int2 focusedCoords;
    const int batchOffset{batch * IMAGES};
    for(int image = 0; image < IMAGES; image++)
    {
        const int gridID{batchOffset + image};
        if constexpr(allFocus)
            focusedCoords = focusCoords(coords, gridID, focusValue);
        else
            focusedCoords = focusCoords(coords, gridID);

        uchar4 px = loadPx(gridID, focusedCoords);
        for(int channel = 0; channel < CHANNELS; channel++)
            destinationPixels[IMAGES * channel + image] = reinterpret_cast<unsigned char * >(&px)[channel];
    }
}

__device__ void pixelsToSharedMemory(int channel, unsigned char *sourcePixels, int warpThreadID, half *currentLocalPixelsMemory, int pixelRowIDInt4)
{
    const int linear = channel * IMAGES;
    int4 packed[2];
    for(int j = 0; j < 8; j++)
    {
        int jj = j << 1;
        reinterpret_cast<half2 *>(&packed)[j] = half2{sourcePixels[linear + jj], sourcePixels[linear + jj + 1]};
    }
    int bankA = warpThreadID % 2;
    int bankB = (warpThreadID + 1) % 2;
    reinterpret_cast<int4 *>(currentLocalPixelsMemory)[pixelRowIDInt4 + bankA] = packed[bankA];
    reinterpret_cast<int4 *>(currentLocalPixelsMemory)[pixelRowIDInt4 + bankB] = packed[bankB];
}

__device__ void storePortionViews(int portion, int2 coords, half *pixels)
{
    for(int viewID = 0; viewID < VIEW_COUNT; viewID++)
    {
        uchar4 color{0, 0, 0, 255};
        for(int channel = 0; channel < CHANNELS; channel++)
            reinterpret_cast<unsigned char * >(&color)[channel] = reinterpret_cast<half * >(pixels)[VIEWS * channel + viewID];
        storePx(color, viewID + portion * VIEW_COUNT, coords);
    }
}

template<bool allFocus>
__global__ void process(half *weights)
{
    using namespace nvcuda;

    auto localWeights = localMemory;
    loadWeightsSync<half>(weights, localWeights);
    int2 coords = getImgCoords();
    if(coordsOutside(coords))
        return;

    const int linearCoords = threadIdx.x + threadIdx.y * blockDim.x;
    const int warpID = linearCoords / WARP_WIDTH;
    const int warpThreadID = linearCoords % WARP_WIDTH;

    const int pixelRowIDInt4{(IMAGES >> 3) *warpThreadID};
    half *currentLocalPixelsMemory = localMemory + weightsSize() + (warpID * PIXEL_MATRIX_SIZE);

    //ROWSxCOLS
    //PIXELSxIMAGES
    wmma::fragment<wmma::matrix_a, PIXELS, VIEWS, IMAGES, half, wmma::row_major> matPixels;
    //IMAGES(weights)xVIEWS
    wmma::fragment<wmma::matrix_b, PIXELS, VIEWS, IMAGES, half, wmma::col_major> matWeights[VIEW_PORTIONS];
    //PIXELSxVIEWS
    wmma::fragment<wmma::accumulator, PIXELS, VIEWS, IMAGES, half> matResult[CHANNELS * VIEW_PORTIONS];
    for(int portion = 0; portion < VIEW_PORTIONS; portion++)
        for(int channel = 0; channel < CHANNELS; channel++)
            wmma::fill_fragment(matResult[channel + portion * CHANNELS], 0.0f);

    uchar4 pixels[IMAGES];
    float focusValue;
    if constexpr(allFocus)
        focusValue = loadFocusFromMap(0, coords);

    const int batchCount{gridSize() >> 4}; // division by IMAGES
    for(int batch = 0; batch < batchCount; batch++)
    {
        loadPixels<allFocus>(batch, coords, reinterpret_cast<unsigned char *>(&pixels), focusValue);
        for(int portion = 0; portion < VIEW_PORTIONS; portion++)
            wmma::load_matrix_sync(matWeights[portion], localWeights + batch * IMAGES + portion * gridSize() * VIEWS, gridSize());

        for(int channel = 0; channel < CHANNELS; channel++)
        {
            pixelsToSharedMemory(channel, reinterpret_cast<unsigned char *>(pixels), warpThreadID, currentLocalPixelsMemory, pixelRowIDInt4);
            wmma::load_matrix_sync(matPixels, currentLocalPixelsMemory, IMAGES);
            for(int portion = 0; portion < VIEW_PORTIONS; portion++)
            {
                const int resultID = channel + portion * CHANNELS;
                wmma::mma_sync(matResult[resultID], matPixels, matWeights[portion], matResult[resultID]);
            }
        }
    }

    for(int portion = 0; portion < VIEW_PORTIONS; portion++)
    {
        for(int channel = 0; channel < CHANNELS; channel++)
        {
            wmma::store_matrix_sync(currentLocalPixelsMemory, matResult[channel + portion * CHANNELS], VIEWS, wmma::mem_row_major);
            reinterpret_cast<int4 *>(pixels)[channel] = reinterpret_cast<int4 *>(currentLocalPixelsMemory)[warpThreadID];
        }
        storePortionViews(portion, coords, reinterpret_cast<half *>(&pixels));
    }

}
}
}
