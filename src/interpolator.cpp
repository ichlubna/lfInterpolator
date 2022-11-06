#include <ranges>
#include <string_view>
#include "lfLoader.h"
#include "interpolator.h"

Interpolator::Interpolator(std::string inputPath) : input{inputPath}
{
    init();
}

void Interpolator::init()
{
    LfLoader lfLoader;
    colsRows = lfLoader.getColsRows();
}

void Interpolator::interpolateClassic(std::string outputPath, std::string trajectory)
{
    interpretTrajectory(trajectory);
}

void interpolateTensor(std::string outputPath, std::string trajectory)
{

}

glm::vec4 Interpolator::interpretTrajectory(std::string trajectory)
{
    constexpr std::string_view delim{","};
    auto numbers = std::views::split(trajectory, delim);

    glm::vec4 absolute;    
    int i{0};
    for (const auto &number : numbers)
    {
        float value = std::stof(std::string{std::string_view(number.begin(), number.end())});
        absolute[i] = value*colsRows[i%2];
        i++;
    }
    return absolute;
}
