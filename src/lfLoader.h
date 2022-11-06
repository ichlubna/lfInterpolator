#include <filesystem>
#include <set>
#include <vector>
#include <glm/glm.hpp>
#include "libs/loadingBar/loadingbar.hpp"

class LfLoader
{
    public:
    using DataGrid = std::vector<std::vector<std::vector<uint8_t>>>;
    glm::uvec2 getColsRows(){return colsRows;}
    void loadData(std::string path);

    private:
    glm::ivec3 resolution;
    DataGrid grid;
    glm::uvec2 colsRows;
    void initGrid(glm::uvec2 inColsRows);
    const std::set<std::filesystem::path> listPath(std::string path) const;
    glm::uvec2 parseFilename(std::string name) const;
    void loadImage(std::string path, glm::uvec2 coords);
};
