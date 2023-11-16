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

    template <typename TT>
    class LocalArray
    {
        public:
        __device__ LocalArray(TT* inData) : data{inData}{}; 
        __device__ TT* ptr(int index)
        {
            return data+index;
        }
        
        template <typename T>
        __device__ T* ptr(int index) 
        {
            return reinterpret_cast<T*>(ptr(index));
        }  

        __device__ TT& ref(int index)
        {
            return *ptr(index);
        }
        
        template <typename T>
        __device__ T& ref(int index)
        {
            return *ptr<T>(index);
        }
     
        TT *data;
    };

    template <typename T>
    class MemoryPartitioner
    {
        public:
        __device__ MemoryPartitioner(T *inMemory)
        {
            memory = inMemory; 
        }

        __device__ LocalArray<T> array(int size)
        {
            T *arr = &(memory[consumed]);
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

    class Indexer
    {
        public:
        __device__ int linearIDBase(int id, int size)
        {
            return linearCoord = id*size;
        } 
        
        __device__ int linearID(int id, int size)
        {
            return linearCoord + id*size;
        }
 
        __device__ int linearCoordsBase(int2 coords, int width)
        {
            return linearCoord = coords.y*width + coords.x;
        }

        __device__ int linearCoords(int2 coords, int width)
        {
            return linearCoord + coords.y*width + coords.x;
        }
        
        static __device__ int linearCoordsSimple(int2 coords, int width)
        {
            return coords.y*width + coords.x;
        }
       
        __device__ int linearCoordsY(int coordY, int width)
        {
            return linearCoord + coordY*width;
        }

        __device__ int getBase()
        {
            return linearCoord;
        }

        private:
        int linearCoord{0};
    };

    template <typename T>
    __device__ static void loadWeightsSync(T *inData, T *data, int size)
    {
        Indexer id;
        id.linearCoordsBase(int2{static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y)}, blockDim.x);
        int i = id.getBase();
        if(i < size)
        {
            int *intLocal = reinterpret_cast<int*>(data);
            int *intIn = reinterpret_cast<int*>(inData);
            intLocal[i] = intIn[i]; 
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
        loadWeightsSync<half>(weights, localWeights.data, weightsSize()/2);  
        PixelArray<float> sum[viewCount()];
        Indexer pxID;
        pxID.linearCoordsBase({coords.x, coords.y}, imgRes().x);
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
                    sum[viewID].addWeighted(localWeights.ref(Indexer::linearCoordsSimple({gridID,viewID}, weightsRes().x)), px);
        }

        for(int viewID=0; viewID<viewCount(); viewID++)
            storePx(sum[viewID].uch4(), viewID, coords);
    }

    __global__ void processTensor(half *weights)
    {
        using namespace nvcuda;

        int2 coords = getImgCoords();
        if(coordsOutside(coords))
            return;

        constexpr int WARP_WIDTH{32};
        int warpID = threadIdx.x/WARP_WIDTH;
        int warpThreadID = threadIdx.x%WARP_WIDTH;
        constexpr int PIXELS{32}, VIEWS{8}, IMAGES{16};

        MemoryPartitioner<half> memoryPartitioner(localMemory);
        auto localWeights = memoryPartitioner.array(weightsSize());
        loadWeightsSync<half>(weights, localWeights.data, weightsSize()/2); 
        auto localInPixels = memoryPartitioner.array(PIXELS*IMAGES);
        auto localOutPixels = memoryPartitioner.array(PIXELS*VIEWS);
        

        //ROWSxCOLS
        //PIXELSxIMAGES
        wmma::fragment<wmma::accumulator, PIXELS, VIEWS, IMAGES, half> matResult;
        //IMAGES(WEIGHTS)xVIEWS
        wmma::fragment<wmma::matrix_a, PIXELS, VIEWS, IMAGES, half, wmma::row_major> matPixels;
        //PIXELSxVIEWS
        wmma::fragment<wmma::matrix_b, PIXELS, VIEWS, IMAGES, half, wmma::col_major> matWeights;
   
        int linearCoords = Indexer::linearCoordsSimple(coords, imgRes().x);
        const int batchCount{gridSize()/IMAGES};
        for(int batch=0; batch<batchCount; batch++)
        {
            const int offset{batch*IMAGES};
            for(int image=offset; image<offset+IMAGES; image++) 
                localInPixels.data[linearCoords] = loadPx<half>(image, coords).channels[0];

            wmma::load_matrix_sync(matPixels, localInPixels.ptr(0), IMAGES);
            wmma::load_matrix_sync(matWeights, localWeights.ptr(batch*IMAGES), gridSize());
            wmma::mma_sync(matResult, matPixels, matWeights, matResult);
        }
        wmma::store_matrix_sync(localOutPixels.ptr(0), matResult, VIEWS, wmma::mem_row_major);
        for(int viewID = 0; viewID<viewCount(); viewID++)
        {
            uchar4 color{0,0,0,0};
            color.x = localOutPixels.data[linearCoords];
            storePx(color, viewID, coords);
        }
 
    }
}
