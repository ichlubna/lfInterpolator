#include <glm/glm.hpp>
#include <cuda_fp16.h>
#include <mma.h>
#include "libs/CudaTensorLibrary/tensor.cu"

namespace Kernels
{
    __device__ constexpr bool GUESS_HANDLES{false};

    __device__ constexpr int CHANNEL_COUNT{4};
    __device__ constexpr int CONSTANTS_COUNT{11};
    __device__ constexpr int VIEW_COUNT{8};
    __constant__ int constants[CONSTANTS_COUNT];
    __device__ int2 imgRes(){return {constants[0], constants[1]};}
    __device__ int2 colsRows(){return{constants[2], constants[3]};}
    __device__ int2 weightsRes(){return {constants[5], constants[4]};}
    __device__ int weightsSize(){return constants[6];}
    __device__ int gridSize(){return constants[5];}
    __device__ int focus(){return constants[7];}
    __device__ int focusRange(){return constants[8];}
    __device__ int2 blockRadius(){return {constants[9], constants[10]};}
    __device__ constexpr int viewCount(){return VIEW_COUNT;}

    __device__ constexpr int MAX_IMAGES{256};
    __device__ constexpr int MAX_SURFACES{256};
    __device__ constexpr int MAP_COUNT{2};
    __constant__ int2 focusedOffsets[MAX_IMAGES];
    __constant__ float2 offsets[MAX_IMAGES];
    __constant__ cudaSurfaceObject_t inputSurfaces[MAX_SURFACES];
    __constant__ cudaSurfaceObject_t outputSurfaces[VIEW_COUNT];
    __constant__ cudaSurfaceObject_t mapSurfaces[MAP_COUNT];
    __device__ constexpr int FOCUS_MAP_IDS_COUNT{32};
    __constant__ int focusMapIDs[FOCUS_MAP_IDS_COUNT];
 
   __device__ int2 focusCoords(int2 coords, int imageID)
    {
        auto offset = focusedOffsets[imageID];
        return {coords.x+offset.x, coords.y+offset.y};
    }
    
    __device__ int2 focusCoords(int2 coords, int imageID, int focus)
    {
        auto offset = offsets[imageID];
        return {static_cast<int>(coords.x+focus*offset.x), static_cast<int>(coords.y+focus*offset.y)};
    }

    extern __shared__ half localMemory[];

    template <typename T>
    class MemoryPartitioner
    {
        public:
        __device__ MemoryPartitioner(T *inMemory)
        {
            memory = inMemory; 
        }

        __device__ T* array(int size)
        {
            T *arr = memory+consumed;
            consumed += size;
            return {arr};
        }
        private:
        T *memory;
        unsigned int consumed{0};
    };

     template <typename T>
        class PixelArray
        {
            public:
            __device__ PixelArray(){};
            __device__ PixelArray(uchar4 pixel) : channels{T(pixel.x), T(pixel.y), T(pixel.z), T(pixel.w)}{};
            __device__ PixelArray(float4 pixel) : channels{pixel.x, pixel.y, pixel.z}{};
            T channels[CHANNEL_COUNT]{0,0,0,0};
            __device__ T& operator[](int index){return channels[index];}
          
             __device__ uchar4 uch4() 
            {
                uchar4 result;
                auto data = reinterpret_cast<unsigned char*>(&result);
                for(int i=0; i<CHANNEL_COUNT; i++)
                    data[i] = __half2int_rn(channels[i]);
                return result;
            }
           
            __device__ void addWeighted(T weight, PixelArray<T> value) 
            {    
                for(int j=0; j<CHANNEL_COUNT; j++)
                    //channels[j] += value[j]*weight;
                    channels[j] = __fmaf_rn(value[j], weight, channels[j]);
            }
            
            __device__ PixelArray<T> operator/= (const T &divisor)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
                    this->channels[j] /= divisor;
                return *this;
            }
            __device__ PixelArray operator-(const PixelArray &value)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
                    this->channels[j] -= value.channels[j];
                return *this;
            }
__device__ PixelArray operator/(const float &value)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
                    this->channels[j] /= value;
                return *this;
            }
  __device__ PixelArray operator+= (const PixelArray &value)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
                    this->channels[j] += value.channels[j];
                return *this;
            }
        };
 
    __device__ int linearCoords(int2 coords, int width)
    {
        return coords.y*width + coords.x;
    }

    template <typename T>
    __device__ static void loadWeightsSync(T *inData, T *data, int size)
    {
        int id = linearCoords(int2{static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y)}, blockDim.x);
        if(id < size)
        {
            int *intLocal = reinterpret_cast<int*>(data);
            int *intIn = reinterpret_cast<int*>(inData);
            intLocal[id] = intIn[id]; 
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
   
    template <typename T>
    __device__ PixelArray<T> loadPx(int imageID, int2 coords)
    {
        constexpr int MULT_FOUR_SHIFT{2};
        if constexpr (GUESS_HANDLES)
            return PixelArray<T>{surf2Dread<uchar4>(imageID+1, coords.x<<MULT_FOUR_SHIFT, coords.y, cudaBoundaryModeClamp)};
        else    
            return PixelArray<T>{surf2Dread<uchar4>(inputSurfaces[imageID], coords.x<<MULT_FOUR_SHIFT, coords.y, cudaBoundaryModeClamp)};
    }
    
    __device__ unsigned char loadPxFromMap(int mapID, int2 coords)
    {
        constexpr int MULT_FOUR_SHIFT{2};
        return surf2Dread<uchar4>(mapSurfaces[mapID], coords.x<<MULT_FOUR_SHIFT, coords.y, cudaBoundaryModeClamp).x;
    }
   
    /* 
    template <typename T>
    __device__ PixelArray<T> loadPx(int imageID, float2 coords, cudaTextureObject_t *textures)
    {
        if constexpr (GUESS_HANDLES)
            return PixelArray<T>{tex2D<uchar4>(imageID+1, coords.x+0.5f, coords.y+0.5f)};
        else    
            return PixelArray<T>{tex2D<uchar4>(textures[imageID], coords.x, coords.y)};
    }
    */

    __device__ void storePx(uchar4 px, int imageID, int2 coords)
    {
        if constexpr (GUESS_HANDLES)
            surf2Dwrite<uchar4>(px, imageID+1+gridSize(), coords.x*sizeof(uchar4), coords.y);
        else    
            surf2Dwrite<uchar4>(px, outputSurfaces[imageID], coords.x*sizeof(uchar4), coords.y);
    }
    
    __device__ void storePxToMap(uchar4 px, int mapID, int2 coords)
    {
            surf2Dwrite<uchar4>(px, mapSurfaces[mapID], coords.x*sizeof(uchar4), coords.y);
    }

    __device__ float distance(PixelArray<float> &a, PixelArray<float> &b)
    {
        return fmaxf(fmaxf(fabsf(a[0]-b[0]), fabsf(a[1]-b[1])), fabsf(a[2]-b[2]));
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
        PixelArray<T> minCol{float4{FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX}};
        PixelArray<T> maxCol{float4{FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN}};
        
        public:
        __device__ void add(PixelArray<T> val)
        {
            minCol[0] = fminf(minCol[0],val[0]);
            minCol[1] = fminf(minCol[1],val[1]);
            minCol[2] = fminf(minCol[2],val[2]);
            maxCol[0] = fmaxf(maxCol[0],val[0]);
            maxCol[1] = fmaxf(maxCol[1],val[1]);
            maxCol[2] = fmaxf(maxCol[2],val[2]);
        }
        __device__ float dispersionAmount()
        {
            return distance(minCol, maxCol); 
        }      
        __device__ ElementRange& operator+=(const PixelArray<T>& rhs){

          add(rhs);
          return *this;
        }
    };

    __device__ float focusDispersion(float focus, int2 coords)
    {
        int2 radius = blockRadius();
        constexpr int BLOCK_DIAMETER{3};
        constexpr int BLOCK_SIZE{BLOCK_DIAMETER*BLOCK_DIAMETER};
        ElementRange<float> dispersions[BLOCK_SIZE];

        for(int viewID= 0; viewID<FOCUS_MAP_IDS_COUNT; viewID++)
        {
            int gridID{focusMapIDs[viewID]};
            int i{0};
            int2 focusedCoords = focusCoords(coords, gridID, focus);
            for(int x = focusedCoords.x-radius.x; x <= focusedCoords.x+radius.x; x+=radius.x) 
                for(int y = focusedCoords.y-radius.y; y <= focusedCoords.y+radius.y; y+=radius.y)
                   dispersions[i++].add(loadPx<float>(gridID, {x,y}));
        }

        float finalDispersion{0};
        for(int i=0; i<BLOCK_SIZE; i++)
            finalDispersion += dispersions[i].dispersionAmount();
        return finalDispersion;
    }

    class MinDispersion
    {
        private:
        float dispersion{FLT_MAX};
        int focus{5};
        public:
        __device__ void add(int newFocus, float newDispersion)
        {
           if(newDispersion < dispersion)
           {
                focus = newFocus;
                dispersion = newDispersion;
           }
        }
        __device__ int getBestFocus()
        {
            return focus;
        } 
    };

    __global__ void estimateFocusMap()
    {
        int2 coords = getImgCoords();
        if(coordsOutside(coords))
            return;

        int step = focusRange()/32;
        MinDispersion minimum;
        for(int f=focus(); f<focus()+focusRange(); f+=step)
           minimum.add(f, focusDispersion(f, coords));

        int bestFocus = minimum.getBestFocus();
        float normalizedFocus = (bestFocus-focus())/static_cast<float>(focusRange());
        unsigned char mapFocus{static_cast<unsigned char>(round(normalizedFocus*UCHAR_MAX))};
        storePxToMap({mapFocus, mapFocus, mapFocus, UCHAR_MAX}, 0, coords);
        //loadPxFromMap(0, coords); 
    }

    template<bool allFocus>
    __global__ void process(half *weights)
    {
        int2 coords = getImgCoords();
        if(coordsOutside(coords))
            return;

        MemoryPartitioner<half> memoryPartitioner(localMemory);
        auto localWeights = memoryPartitioner.array(weightsSize());
        loadWeightsSync<half>(weights, localWeights, weightsSize()/2);  
        PixelArray<float> sum[viewCount()];
        
        int2 focusedCoords;
        int focusValue;
        if constexpr (allFocus)
            focusValue = round((static_cast<float>(loadPxFromMap(0, coords))/UCHAR_MAX)*focusRange());

        for(int gridID = 0; gridID<gridSize(); gridID++)
        { 
            if constexpr (allFocus)
                focusedCoords = focusCoords(coords, gridID, focusValue);
            else
                focusedCoords = focusCoords(coords, gridID);

            auto px{loadPx<float>(gridID, focusedCoords)};
            for(int viewID=0; viewID<viewCount(); viewID++)
                    sum[viewID].addWeighted(localWeights[linearCoords({gridID,viewID}, weightsRes().x)], px);
        }

        for(int viewID=0; viewID<viewCount(); viewID++)
            storePx(sum[viewID].uch4(), viewID, coords);
    }

    __device__ half clamp(half value, float minimum, float maximum)
    {
        return max(min(value, maximum), minimum);
    }

    template<bool allFocus>
    __global__ void processTensor(half *weights)
    {
        using namespace nvcuda;

        int2 coords = getImgCoords();
        if(coordsOutside(coords))
            return;

        const int linearCoords = threadIdx.x+threadIdx.y*blockDim.x;
        constexpr int WARP_WIDTH{32};
        const int warpID = linearCoords/WARP_WIDTH;
        const int warpThreadID = linearCoords%WARP_WIDTH;
        //const int warpCount = blockDim.x*blockDim.y/WARP_WIDTH;
        constexpr int WARP_COUNT = 256/WARP_WIDTH;
        constexpr int PIXELS{32}, VIEWS{8}, IMAGES{16};

        MemoryPartitioner<half> memoryPartitioner(localMemory);
        auto localWeights = memoryPartitioner.array(weightsSize());
        loadWeightsSync<half>(weights, localWeights, weightsSize()/2); 
        
        constexpr int CHANNELS{3};
        constexpr int PIXEL_MATRIX_SIZE{PIXELS*IMAGES};
        constexpr int RESULT_MATRIX_SIZE{PIXELS*VIEWS};
        auto localInPixels = memoryPartitioner.array(PIXEL_MATRIX_SIZE*WARP_COUNT*CHANNELS);
        auto localOutPixels = memoryPartitioner.array(RESULT_MATRIX_SIZE*WARP_COUNT*CHANNELS); 

        //ROWSxCOLS
        //PIXELSxIMAGES
        wmma::fragment<wmma::matrix_a, PIXELS, VIEWS, IMAGES, half, wmma::row_major> matPixels;
        //IMAGES(weights)xVIEWS
        wmma::fragment<wmma::matrix_b, PIXELS, VIEWS, IMAGES, half, wmma::col_major> matWeights;
        //PIXELSxVIEWS
        wmma::fragment<wmma::accumulator, PIXELS, VIEWS, IMAGES, half> matResult[CHANNELS];
        for(int channel=0; channel<CHANNELS; channel++) 
            wmma::fill_fragment(matResult[channel], 0.0f);

        const int pixelRowID{IMAGES*warpThreadID};
        half *currentLocalInPixels[CHANNELS];
        currentLocalInPixels[0] = localInPixels+(warpID*PIXEL_MATRIX_SIZE);
        currentLocalInPixels[1] = localInPixels+(PIXEL_MATRIX_SIZE*(warpID+WARP_COUNT));
        currentLocalInPixels[2] = localInPixels+(PIXEL_MATRIX_SIZE*(warpID+WARP_COUNT*2));
        half *currentLocalOutPixels[CHANNELS];
        currentLocalOutPixels[0] = localOutPixels+(warpID*RESULT_MATRIX_SIZE);
        currentLocalOutPixels[1] = localOutPixels+(RESULT_MATRIX_SIZE*(warpID+WARP_COUNT));
        currentLocalOutPixels[2] = localOutPixels+(RESULT_MATRIX_SIZE*(warpID+WARP_COUNT*2));
       
        int2 focusedCoords;
        int focusValue; 
        if constexpr (allFocus)
            focusValue = round((static_cast<float>(loadPxFromMap(0, coords))/UCHAR_MAX)*focusRange());

        const int batchCount{gridSize()/IMAGES};
        for(int batch=0; batch<batchCount; batch++)
        {
            const int offset{batch*IMAGES};
            for(int image=0; image<IMAGES; image++) 
            {
                const int gridID{offset+image};
                if constexpr (allFocus)
                    focusedCoords = focusCoords(coords, gridID, focusValue);
                else
                    focusedCoords = focusCoords(coords, gridID);

                auto pixel = loadPx<half>(gridID, focusedCoords);
                currentLocalInPixels[0][pixelRowID+image] = pixel.channels[0];
                currentLocalInPixels[1][pixelRowID+image] = pixel.channels[1];
                currentLocalInPixels[2][pixelRowID+image] = pixel.channels[2];
            }
            wmma::load_matrix_sync(matWeights, localWeights+batch*IMAGES, gridSize());
            for(int channel=0; channel<CHANNELS; channel++) 
            {
                wmma::load_matrix_sync(matPixels, currentLocalInPixels[channel], IMAGES);
                wmma::mma_sync(matResult[channel], matPixels, matWeights, matResult[channel]);
            }
        }
        for(int channel=0; channel<CHANNELS; channel++) 
            wmma::store_matrix_sync(currentLocalOutPixels[channel], matResult[channel], VIEWS, wmma::mem_row_major);
        const int viewRowID{VIEWS*warpThreadID};
        for(int viewID = 0; viewID<viewCount(); viewID++)
        {
            uchar4 color{0,0,0,255};
            /*color.x = clamp(currentLocalOutPixels[0][viewID+viewRowID], 0, 255);
            color.y = clamp(currentLocalOutPixels[1][viewID+viewRowID], 0, 255);
            color.z = clamp(currentLocalOutPixels[2][viewID+viewRowID], 0, 255);*/
            color.x = currentLocalOutPixels[0][viewID+viewRowID];
            color.y = currentLocalOutPixels[1][viewID+viewRowID];
            color.z = currentLocalOutPixels[2][viewID+viewRowID];
            storePx(color, viewID, coords);
        }
 
    }
}
