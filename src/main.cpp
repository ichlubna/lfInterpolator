#include "interpolator.h"
#include "libs/arguments/arguments.hpp"

int main(int argc, char **argv)
{
    Arguments args(argc, argv);
    std::string path = static_cast<std::string>(args["-i"]);
    std::string trajectory = static_cast<std::string>(args["-t"]);
    std::string outputPath = static_cast<std::string>(args["-o"]);
    float focus = args["-f"];
    float range = args["-r"];
    std::string method = static_cast<std::string>(args["-m"]);

    std::string helpText{ "Usage:\n"
                          "Example: lfInterpolator -i /MyAmazingMachine/thoseImages -t 0.0,0.0,1.0,1.0  -o ./outputs\n"
                          "-o - output path\n"
                          "-i - folder with lf grid images - named as column_row.extension, e.g. 01_12.jpg\n"
                          "-t - trajectory of the camera in normalized coordinates of the grid format: startCol,startRow,endCol,endRow\n"
                          "-s - the amount of the spatial 3D effect - affects how much are views close to the virtual one prioritized (default=3.0)\n"
                          "-a - aspect ratio of the spacing of the capturing cameras in the grid (horizontal/vertical space) (default=1)\n"
                          "-m - interpolation method:\n"
                          "     STD - standard interpolation kernel\n"
                          "     TEN_WM - WMMA tensor cores\n"
                          "     TEN_OP - optimized memory tensor cores\n"
                          "The following arguments are normalized offsets of the images in shift & sum\n"
                          "-f - focusing value (default=0)\n"
                          "-r - focusing range (will be added to the focusing value) - will produce all-focused result if used\n"
                        };
    if(args.printHelpIfPresent(helpText))
        return 0;

    float effect = static_cast<float>(args["-s"]);
    if(effect <= 0)
        effect = 3;
    
    float aspect = static_cast<float>(args["-a"]);
    if(aspect <= 0)
        aspect = 1;

    if(!args["-i"] || !args["-t"] || !args["-o"] || !args["-m"])
    {
        std::cerr << "Missing required parameters. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        Interpolator interpolator(path);
        interpolator.interpolate(outputPath, trajectory, focus, range, method, effect, aspect);
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
