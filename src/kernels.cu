#include <glm/glm.hpp>
#include <cuda_fp16.h>

namespace Kernels
{
    __device__ constexpr bool GUESS_HANDLES{false};

    __device__ constexpr int CHANNEL_COUNT{4};
    __device__ constexpr int CONSTANTS_COUNT{7};
    __device__ constexpr int VIEW_COUNT{8};
    __constant__ int constants[CONSTANTS_COUNT];
    __device__ int2 imgRes(){return {constants[0], constants[1]};}
    __device__ int2 colsRows(){return{constants[2], constants[3]};}
    __device__ int2 weightsRes(){return {constants[4], constants[5]};}
    __device__ int weightsSize(){return constants[6];}
    __device__ int gridSize(){return constants[5];}
    __device__ constexpr int viewCount(){return VIEW_COUNT;}

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
                    //sum[j] += fPixel[j]*weight;
                    channels[j] = __fmaf_rn(value[j], weight, channels[j]);
            }
            
            __device__ PixelArray<T> operator/= (const T &divisor)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
                    this->channels[j] /= divisor;
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
        //TODO more than threads?
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
   
    template <typename T>
    __device__ PixelArray<T> loadPx(int imageID, int2 coords, cudaTextureObject_t *textures)
    {
        if constexpr (GUESS_HANDLES)
            return PixelArray<T>{tex2D<uchar4>(textures[imageID], coords.x+0.5f, coords.y+0.5f)};
        else    
            return PixelArray<T>{tex2D<uchar4>(imageID, coords.x+0.5f, coords.y+0.5f)};
    }

    __device__ void storePx(uchar4 px, int imageID, int2 coords, cudaSurfaceObject_t *surfaces)
    {
        if constexpr (GUESS_HANDLES)
            surf2Dwrite<uchar4>(px, imageID*viewCount(), coords.x*sizeof(uchar4), coords.y);
        else    
            surf2Dwrite<uchar4>(px, surfaces[imageID], coords.x*sizeof(uchar4), coords.y);
    }

    __global__ void process(cudaTextureObject_t *textures, cudaSurfaceObject_t *surfaces, half *weights)
    {
        int2 coords = getImgCoords();
        if(coordsOutside(coords))
            return;

        MemoryPartitioner<half> memoryPartitioner(localMemory);
        auto localWeights = memoryPartitioner.array(weightsSize());
        loadWeightsSync<half>(weights, localWeights.data, weightsSize()/2);  
        Indexer weightMatIndex;
        PixelArray<float> sum[viewCount()];
        Indexer pxID;
        pxID.linearCoordsBase({coords.x, coords.y}, imgRes().x);
        for(int gridID = 0; gridID<gridSize(); gridID++)
        {
            auto px{loadPx<float>(gridID, coords, textures)};
            for(int viewID=0; viewID<viewCount(); viewID++)
                    sum[viewID].addWeighted(localWeights.ref(weightMatIndex.linearCoords({gridID,viewID}, weightsRes().x)), px);
        }

        for(int viewID=0; viewID<viewCount(); viewID++)
            storePx(sum[viewID].uch4(), viewID, coords, surfaces);
    }

    __global__ void processTensor(cudaTextureObject_t *textures, cudaSurfaceObject_t *surfaces, half *weights)
    {

    }


}
