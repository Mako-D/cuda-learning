cmake_minimum_required (VERSION 3.8)
#=============

set(PROJECT_NUM_FOLDER "01/real2complex")

#=============
# For target library compability
cmake_policy(SET CMP0079 NEW)

#=============
# Project name
project ("CUDAPractice" LANGUAGES CUDA CXX)

#=============
# Settings for CUDA
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe")
set(CMAKE_BUILD_TYPE Debug)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -g -G)  # enable cuda-gdb
endif()
if(MSVC)
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()

#=============
# Add CUDA toolkit
if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED)
endif()

#=============
# Linker settings
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

#=============
# Cmake path
set(CMAKE_PREFIX_PATH "E:/Program Files/Qt/6.2.4/msvc2019_64/lib/cmake")

#=============
# Add Qt library
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

include_directories(${QT_INCLUDES})

#=============
# Add project files from dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_subdirectory(${PROJECT_NUM_FOLDER})

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -std=c++17
                       -Xcompiler
                       -g
                       -G
                       "/std:c++17"
                        >)
#=============
# Link library
target_link_libraries(${PROJECT_NAME} PUBLIC Qt${QT_VERSION_MAJOR}::Widgets CUDA::cufft)

#=============
# Compile CUDA files separatly
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#=============