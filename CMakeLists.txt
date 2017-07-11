cmake_minimum_required(VERSION 3.7)
project(SSLAM)

IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(OpenCV_DIR "/opt/opencv2/share/OpenCV")

find_package(OpenCV 2.4.10 REQUIRED)
set(OpenCV_INCLUDE_DIRS "/opt/opencv2/include/opencv" "/opt/opencv2/include")
message(STATUS "${OpenCV_INCLUDE_DIRS}")
#find_package(OpenCV 3.0 QUIET)
#if (NOT OpenCV_FOUND)
#    find_package(OpenCV 2.4.3 QUIET)
#    if (NOT OpenCV_FOUND)
#        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#    endif ()
#endif ()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Glog)
find_package(Ceres REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Ceres_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
        src/Monitor.cpp
        src/Main.cpp
        src/GlobalParameters.cpp
        src/ORBextractor.cpp
        src/ORBmatcher.cpp
        src/Frame.cpp
        src/Preprocessor.cpp
        src/MapPoint.cpp
        src/Map.cpp
        src/Tracker.cpp
        src/Optimizer.cpp
        src/Converter.cpp
        src/KeyFrame.cpp
        src/EpipolarTriangle.cpp
        src/FrameDrawer.cpp
        src/MapDrawer.cpp
        src/Viewer.cpp
        src/ETFSLAM.cpp
        src/TestTriangleForMatching.cpp src/LocalMapper.cpp)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        glog
        ${CERES_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        )


