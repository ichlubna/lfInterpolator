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
    auto image = std::make_unique<Image>();
    image->pixels = stbi_load(path.c_str(), &image->width, &image->height, &image->channels, STBI_rgb_alpha);
    if(image->pixels == nullptr)
        throw std::runtime_error("Cannot load image " + path);
    size_t size = image->width * image->height * 4; //image->channels;
    resolution = {image->width, image->height};
    channels = image->channels;

    dataGrid[coords.x][coords.y].resize(size);
    memcpy(dataGrid[coords.x][coords.y].data(), image->pixels, size);
}

