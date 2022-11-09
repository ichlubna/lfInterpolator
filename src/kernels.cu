#include <glm/glm.hpp>
#include <cuda_fp16.h>

namespace Kernels
{
    __device__ constexpr int WEIGHTS_COLS{0};
    __device__ constexpr int WEIGHTS_ROWS{0};
    __device__ constexpr int CHANNEL_COUNT{4};
    __device__ constexpr int OUT_VIEWS_COUNT{8};

    __device__ constexpr int CONSTANTS_COUNT{4};
    __constant__ int constants[CONSTANTS_COUNT];
    __device__ int2 imgRes(){return {constants[0], constants[1]};}
    __device__ int2 colsRows(){return {constants[2], constants[3]};}

    extern __shared__ half localMemory[];


    __device__ bool coordsOutside(int2 coords)
    {
        int2 resolution = imgRes();
        if(coords.x >= resolution.x || coords.y >= resolution.y)
            return false;
    }

    __device__ int2 getImgCoords()
    {
        int2 coords;
        coords.x = (threadIdx.x + blockIdx.x * blockDim.x);
        coords.y = (threadIdx.y + blockIdx.y * blockDim.y);
        return coords;
    }

    __global__ void process(cudaTextureObject_t *textures, cudaSurfaceObject_t *surfaces, half *weights)
    {
        int2 coords = getImgCoords();
        if(coordsOutside(coords))
            return;

        //auto px = tex2D<uchar4>(textures[0], coords.x+0.5f, coords.y+0.5f);
        uchar4 px{255,0,255,255};
        surf2Dwrite<uchar4>(px, surfaces[0], coords.x, coords.y, cudaBoundaryModeTrap);
        
    }

}
