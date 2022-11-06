#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <cstring>
#include "lfLoader.h"

const std::set<std::filesystem::path> LfLoader::listPath(std::string path) const
{
    auto dir = std::filesystem::directory_iterator(path);
    std::set<std::filesystem::path> sorted;
    for(const auto &file : dir)
        sorted.insert(file.path().filename());
    return sorted;
}

glm::uvec2 LfLoader::parseFilename(std::string name) const
{
    int delimiterPos = name.find('_');
    int extensionPos = name.find('.');
    auto row = name.substr(0, delimiterPos);
    auto col = name.substr(delimiterPos + 1, extensionPos - delimiterPos - 1);
    return {stoi(row), stoi(col)};
}

void LfLoader::loadImage(std::string path, glm::uvec2 coords)
{
    uint8_t *pixels = stbi_load(path.c_str(), &resolution.x, &resolution.y, &resolution.z, STBI_rgb_alpha);
    if(pixels == nullptr)
        throw std::runtime_error("Cannot load image " + path);
    size_t size = glm::dot(glm::vec3(resolution), glm::vec3(1));
    grid[coords.x][coords.y].resize(size);
    std::memcpy(grid[coords.x][coords.y].data(), pixels, size);
}

void LfLoader::initGrid(glm::uvec2 inColsRows)
{
    colsRows = inColsRows;
    grid.resize(colsRows.x);
    for(auto &row : grid)
        row.resize(colsRows.y);
}

void LfLoader::loadData(std::string path)
{
    auto files = listPath(path);
    initGrid(parseFilename(*files.rbegin()));
    
    for(auto const &file : files)
    {
        auto coords = parseFilename(file);
        loadImage(path+"/"+file.string(), coords);   
    }
}
