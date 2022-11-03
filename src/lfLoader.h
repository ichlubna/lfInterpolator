#include <filesystem>
#include <set>
#include <vector>
#include <glm/glm.hpp>
#include "libs/loadingBar/loadingbar.hpp"
#include "libs/stb_image.h"

class LfLoader
{
    public:
    using DataGrid = std::vector<std::vector<std::vector<uint8_t>>>;

    private:
    const std::set<std::filesystem::path> listPath(std::string path) const;
    glm::uvec2 parseFilename(std::string name) const;
    void loadImage(std::string path, glm::uvec2 coords);
};
