#include <vector>
#include <glm/glm.hpp>
#include <string>

class Interpolator
{
    public:
    Interpolator(std::string inputPath);
    ~Interpolator();
    void interpolate(std::string outputPath, std::string trajectory, float focus, bool tensor);

    private:
    size_t kernelBenchmarkRuns{10};
    std::vector<int*> outputArrays;
    void *surfaceObjectsArr;
    void *textureObjectsArr;
    int *weights;
    size_t channels{4};
    int viewCount{8};
    size_t sharedSize{0};
    glm::ivec2 colsRows;
    glm::ivec3 resolution;
    std::string input;
    void init();
    void loadGPUData();
    void loadGPUOffsets(float focus);
    void loadGPUConstants();
    void loadGPUWeights(glm::vec4 startEndPoints);
    void storeResults(std::string path);
    std::vector<float> generateWeights(glm::vec2 coords);
    std::vector<glm::vec2> generateTrajectory(glm::vec4 startEndPoints);
    glm::vec4 interpretTrajectory(std::string trajectory);    
    std::pair<int, int*> createSurfaceObject(glm::ivec3 size);
    int createTextureObject(const uint8_t *data, glm::ivec3 size);
};
