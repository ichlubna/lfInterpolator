#include <glm/glm.hpp>
#include <string>

class Interpolator
{
    public:
    Interpolator(std::string inputPath);
    void interpolateTensor(std::string outputPath, std::string trajectory);
    void interpolateClassic(std::string outputPath, std::string trajectory);

    private:
    glm::uvec2 colsRows;
    std::string input;
    void init();
    glm::vec4 interpretTrajectory(std::string trajectory);
};
