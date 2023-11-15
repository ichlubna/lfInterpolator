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

    std::string helpText{ "Usage:\n"
                          "Example: lfInterpolator -i /MyAmazingMachine/thoseImages -t 0.0,0.0,1.0,1.0  -o ./outputs\n"
                          "-o - output path\n"
                          "-i - folder with lf grid images - named as column_row.extension, e.g. 01_12.jpg\n"
                          "-t - trajectory of the camera in normalized coordinates of the grid format: startCol,startRow,endCol,endRow\n"
                          "The following arguments are pixel offsets of the images in shift & sum\n"
                          "-f - focusing value\n"
                          "-r - focusing range (will be added to the focusing value) - will produce all-focused result if used\n"
                          "--tensor - tensor cores are used when present\n"
                        };
    if(args.printHelpIfPresent(helpText))
        return 0;

    if(!args["-i"] || !args["-t"] || !args["-o"])
    {
        std::cerr << "Missing required parameters. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        Interpolator interpolator(path);
        interpolator.interpolate(outputPath, trajectory, focus, range, args["--tensor"]);
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
