set(HEADER_FILES
        inc/fft.h)

set(SOURCE_FILES
        src/fft.cpp)
include_directories(inc)
add_library(fft_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})