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
    __device__ int gridSize(){return constants[2]*constants[3];}

    extern __shared__ half localMemory[];

    template <typename TT>
    class Matrix
    {
        public:
        __device__ Matrix(TT* inData) : data{inData}{}; 
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
     
        half *data;
    };

    template <typename TT>
    class MemoryPartitioner
    {
        public:
        __device__ MemoryPartitioner(TT *inMemory)
        {
            memory = inMemory; 
        }

        __device__ Matrix<TT> getMatrix(int count, int rows, int cols)
        {
            int size = rows*cols*count;
            TT *arr = &(memory[consumed]);
            consumed += size;
            return {arr};
        }
        private:
        TT *memory;
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
          
             __device__ uchar4 getUchar4() 
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
        if(threadIdx.x < size)
        {
            int *intLocal = reinterpret_cast<int*>(data);
            int *intIn = reinterpret_cast<int*>(inData);
            intLocal[threadIdx.x] = intIn[threadIdx.x]; 
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

    __global__ void process(cudaTextureObject_t *textures, cudaSurfaceObject_t *surfaces, half *weights)
    {
        int2 coords = getImgCoords();
        if(coordsOutside(coords))
            return;

        MemoryPartitioner<half> memoryPartitioner(localMemory);
        auto localWeights = memoryPartitioner.getMatrix(1, WEIGHTS_ROWS, WEIGHTS_COLS);
        loadWeightsSync<half>(weights, localWeights.data, WEIGHTS_COLS*WEIGHTS_ROWS/2);  
        Indexer weightMatIndex;
        PixelArray<float> sum[OUT_VIEWS_COUNT];

        Indexer pxID;
        pxID.linearCoordsBase({coords.x, coords.y}, imgRes().x);
        for(int gridID = 0; gridID<gridSize(); gridID++)
        {
            PixelArray<float> px{tex2D<uchar4>(textures[gridID], coords.x+0.5f, coords.y+0.5f)};
            for(int viewID=0; viewID<OUT_VIEWS_COUNT; viewID++)
            {
                    int x = viewID;
                    int y = gridID;
                    //sum[i].addWeighted(localWeights.ref(weightMatIndex.linearCoords({x,y}, WEIGHTS_COLS)), px);
                    sum[viewID].addWeighted(0.25, px);
            }
        }

        for(int i=0; i<OUT_VIEWS_COUNT; i++)
            surf2Dwrite<uchar4>(sum[i].getUchar4(), surfaces[i], coords.x*sizeof(uchar4), coords.y);
    }

}
