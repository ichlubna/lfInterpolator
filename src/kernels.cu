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


}
