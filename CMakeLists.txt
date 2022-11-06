cmake_minimum_required(VERSION 3.1)
project(lfInterpolator)
set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules")

set(SOURCES 
src/lfLoader.cpp
src/main.cpp
src/interpolator.cpp
src/libs/arguments/arguments.hpp
src/libs/loadingBar/loadingbar.hpp
src/libs/stb_image.h
src/libs/stb_image_write.h
)

find_package(glm REQUIRED)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/src/libs)

add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(${PROJECT_NAME} glm)
