cmake_minimum_required(VERSION 3.7)
project(woods)

include_directories(src/)
set(SOURCE_FILES src/python.cpp
                 src/interface.hpp
                 src/check.hpp
                 src/utils.hpp
                 src/tree/tree.hpp
                 src/tree/impurity.hpp
                 src/tree/split.hpp
                 src/tree/rule.hpp
                 src/ensemble/boosting.hpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (WIN32)
    set(CMAKE_CXX_FLAGS "/O2 /EHsc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT")
else()
    set(CMAKE_CXX_FLAGS "-O2")
endif()

set (Boost_COMPILER "-vc")
set (BOOST_ROOT "C:/Boost")
set (BOOST_INCLUDEDIR  "C:/Users/andru/Anaconda3/Library/include")
set (BOOST_LIBRARYDIR  "C:/Users/andru/Anaconda3/Library/lib")
## set (BOOST_INCLUDEDIR  "C:/Boost/include/boost-1_73/")
## set (BOOST_LIBRARYDIR  "C:/Boost/lib")
set (Boost_USE_STATIC_RUNTIME ON)  
# set (Boost_DEBUG ON)
# set(Boost_DETAILED_FAILURE_MSG ON)
set(PY_MAJOR_VERSION "3")
set(PY_MINOR_VERSION "7")

# find_package(Boost 1.68 COMPONENTS python37)

set(PYTHON_INCLUDE_PATH "C:/Users/andru/Anaconda3/include")
set(PYTHON_LIBRARIES "C:/Users/andru/Anaconda3/libs/") # python37.lib

include_directories(${PYTHON_INCLUDE_PATH})
link_directories(${PYTHON_LIBRARIES})

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(Boost REQUIRED COMPONENTS python37 numpy37 random)
FIND_PACKAGE(PythonInterp 3)
FIND_PACKAGE(PythonLibs 3 REQUIRED)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    # add_executable(woods ${SOURCE_FILES})
    # add_library(woods MODULE ${SOURCE_FILES})
    PYTHON_ADD_MODULE(woods ${SOURCE_FILES})
    target_link_libraries(woods ${Boost_LIBRARIES})
    # target_link_libraries(woods Boost::python37)
    message(STATUS "Boost is loaded")
else()
    message(STATUS "Something is wrong with boost")
endif()