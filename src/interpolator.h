#include <vector>
#include <glm/glm.hpp>
#include <string>

class Interpolator
{
    public:
    Interpolator(std::string inputPath);
    ~Interpolator();
    void interpolate(std::string outputPath, std::string trajectory, float focus, float range, std::string method, float effect);

    private:
    size_t kernelBenchmarkRuns{100};
    std::vector<int*> surfaceInputArrays;
    std::vector<int*> surfaceOutputArrays;
    int *weights;
    int focus{0};
    int range{0};
    size_t channels{4};
    size_t sharedSize{0};
    glm::ivec2 colsRows;
    glm::ivec3 resolution;
    std::string input;
    void init();
    void loadGPUData();
    void loadGPUOffsets();
    void loadGPUConstants();
    void loadGPUWeights(glm::vec4 startEndPoints, float effect);
    void selectFocusMapViews(glm::vec4 startEndPoints);
    int* loadImageToArray(const uint8_t *data, glm::ivec3 size);
    void storeResults(std::string path);
    std::vector<float> generateWeights(glm::vec2 coords, float effect);
    std::vector<glm::vec2> generateTrajectory(glm::vec4 startEndPoints);
    glm::vec4 interpretTrajectory(std::string trajectory);    
    std::pair<int, int*> createSurfaceObject(glm::ivec3 size, const uint8_t *data=nullptr);
    int createTextureObject(glm::ivec3 size, const uint8_t *data);
};
