#include <math.h>
#include <complex>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

const double PI = 3.141592653589793238460;
class fft {
public:
    // Constructor
    fft();
    // Destructor
    virtual ~fft();

    /**
     * FFT
     * This takes a vector (think an array) of complex types.
     *
     * The Complex Data Type looks like:
     * ( 0 , 0i), where the first term is the real part, and the second term is the imaginary part
     *
     * We're declaring a complex type of doubles, so we can have:
     * ( 0.0000, 0.0000i ), instead of just Integers, which gives us more accuracy.
     *
     * So, the vector we are passing looks like:
     * [ ( a , bi ), ... , ( y , zi ) ]
     * */
    void fast_fourier_transform(std::vector<std::complex<double>> &x);
    // Inverse FFT
    void inverse_fft(std::vector<std::complex<double>> &x);

    /**
     * Loads an image from the given filename into the Mat at location
     *
     * returns:
     *      1 - success
     *      0 - failure
     * */

    int load_image(char * file_name, cv::Mat &location);

    void display_image(cv::Mat image);
    int save_image(char * file_name, cv::Mat image);

private:
};