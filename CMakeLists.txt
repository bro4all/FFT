cmake_minimum_required(VERSION 3.9)
project(fft)
find_package(OpenCV)

if(WIN32)
    if (CYGWIN)
        add_definitions(-std=gnu++0x)
    elseif(MINGW)
        add_definitions(-std=c++11)
    endif (CYGWIN)
endif(WIN32)
if(APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
    add_definitions(-D__GLIBCXX__)
endif(APPLE)
if(UNIX)
    add_definitions(-std=gnu++0x)
endif(UNIX)

set(SOURCE_FILES main.cpp lib/fft/src/fft.cpp)

add_executable(labs_main ${SOURCE_FILES})

include_directories(lib/fft/inc ${OpenCV_INCLUDE_DIRS})

add_subdirectory(lib/fft fft_lib)

add_subdirectory(test)

target_link_libraries(labs_main fft_lib ${OpenCV_LIBRARIES})