project(fft_Tests)

find_package(OpenCV)
IF (OpenCV_FOUND)
    message("OpenCV Found...")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
    message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
endif()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(runBasicTests
        fft_tests.cpp)

add_subdirectory(gtest-1.7.0)

target_link_libraries(runBasicTests gtest gtest_main)
target_link_libraries(runBasicTests fft_lib)
target_link_libraries(runBasicTests fft_lib ${OpenCV_LIBRARIES})