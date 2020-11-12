# ----------------------------------------------------------------------
# 2020 Bernd Pfrommer bernd.pfrommer@gmail.com
#
cmake_minimum_required(VERSION 3.9)
project(vilib VERSION 1.0.0 DESCRIPTION "fast cuda fast feature tracking" LANGUAGES CXX CUDA)

include (CMakePrintHelpers)
include (CMakePackageConfigHelpers)
include (CPack)
include (CheckLanguage)
include (GNUInstallDirs)

find_package(OpenCV REQUIRED COMPONENTS core)
find_package(Eigen3 REQUIRED)

check_language(CUDA)
option(BUILD_SHARED_LIBS "build shared libraries" ON)
cmake_print_variables(BUILD_SHARED_LIBS)


set(CUSTOM_OPENCV_PATH "" CACHE PATH "path to custom opencv version")

add_library(vilib SHARED
  visual_lib/src/cuda_common.cpp
  visual_lib/src/timer.cpp
  visual_lib/src/common/framebundle.cpp
  visual_lib/src/common/frame.cpp
  visual_lib/src/common/point.cpp
  visual_lib/src/feature_tracker/feature_tracker_base.cpp
  visual_lib/src/feature_tracker/feature_tracker_gpu.cpp
  visual_lib/src/storage/ros.cpp
  visual_lib/src/storage/opencv.cpp
  visual_lib/src/storage/pyramid_pool.cpp
  visual_lib/src/storage/subframe.cpp
  visual_lib/src/storage/subframe_pool.cpp
  visual_lib/src/preprocess/pyramid_cpu.cpp
  visual_lib/src/preprocess/image_preprocessing.cpp
  visual_lib/src/statistics.cpp
  visual_lib/src/feature_detection/fast/opencv/fast_cpu.cpp
  visual_lib/src/feature_detection/fast/rosten/fast_cpu.cpp
  visual_lib/src/feature_detection/fast/rosten/fast_9.cpp
  visual_lib/src/feature_detection/fast/rosten/fast_12.cpp
  visual_lib/src/feature_detection/fast/rosten/fast_10.cpp
  visual_lib/src/feature_detection/fast/rosten/nonmax.cpp
  visual_lib/src/feature_detection/fast/rosten/fast_11.cpp
  visual_lib/src/feature_detection/fast/rosten/fast.cpp
  visual_lib/src/feature_detection/fast/fast_gpu.cpp
  visual_lib/src/feature_detection/detector_base_gpu.cpp
  visual_lib/src/feature_detection/detector_benchmark.cpp
  visual_lib/src/feature_detection/detector_base.cpp
  visual_lib/src/timergpu.cpp
  visual_lib/src/feature_tracker/feature_tracker_cuda_tools.cu
  visual_lib/src/preprocess/pyramid_gpu.cu
  visual_lib/src/feature_detection/fast/fast_gpu_cuda_tools.cu
  visual_lib/src/feature_detection/detector_base_gpu_cuda_tools.cu
  )
#  visual_lib/src/cuda_common.cpp

# the PUBLIC keyword means that any targets linking to this
# target also need that include directory. 
target_include_directories(vilib PUBLIC  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/visual_lib/include>  
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

# will require that the compiler support compiler standard, and
# add the flag if necessary. The PUBLIC keyword means any targets depending
# on this target will also have to satisfy that requirement
target_compile_features(vilib PUBLIC cxx_std_11)

# tell cmake to use e.g. -std=c++11 instead of -std=g++11, and bomb
# out if the standard is not required
set_target_properties(vilib
  PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS OFF
  LINK_WHAT_YOU_USE OFF         # switch to on to check for unnecessary links
  POSITION_INDEPENDENT_CODE ON
  )

target_link_libraries(vilib PRIVATE
  opencv_core Eigen3::Eigen)

# Set different compiler options for cxx and nvcc

set(cxx_options -Wall -Wextra -Werror -Wfatal-errors -ffast-math -fsee -fno-signed-zeros -fno-math-errno -funroll-loops -fno-finite-math-only -march=native -O3 -DNDEBUG)
set(nvcc_options --device-w --ftz=true --prec-div=false --prec-sqrt=false --fmad=true --default-stream per-thread -O3 -DNDEBUG)

# the generator is only expanded if <COMPILE_LANGUAGE:FOO> evaluates to true
target_compile_options(vilib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${cxx_options}> $<$<COMPILE_LANGUAGE:CUDA>:${nvcc_options}>)

# Specify how to install the target.
# The EXPORT keyword associates the installed target files with an "export" called
# ${PROJECT_NAME}Targets. The file will later be generated when later install(EXPORT) is
# called on that target.
# Note that this install does not install the header files yet.
install(TARGETS vilib
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION include
  )

install(DIRECTORY visual_lib/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


# generate the ConfigVersion.cmake file that will be included by Config.cmake
write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion
  )

# generate the Config.cmake file
configure_file(cmake/${PROJECT_NAME}Config.cmake.in ${PROJECT_NAME}Config.cmake @ONLY)

# generate the Targets.cmake file from the export info learned during the
# installation of the vilib target
install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

# install the generated version and config files
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
        )


# Packaging support
set(CPACK_PACKAGE_VENDOR "vilib")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "fast cuda based FAST feature lib")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENCE.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
