#include <math.h>
#include <complex>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

const double PI = 3.141592653589793238460;
struct WINDOW_SIZE_SM {
    int height = 250;
    int width = 250;
};

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
    void fast_fourier_transform(std::vector<std::complex<double>> &c);
    // Inverse fft1d
    void inverse_fft(std::vector<std::complex<double>> &c);

    /**
     * fft2d:
     * Fast Fourier Transform in 2 Dimensions.
     * @param complex_image - array of complex<double> vectors containing the image data.
     * @param rows - the number of rows in complex_image
     * @param cols - number of columns in complex_image
     * @param direction - can be 1 for FFT, -1 for IFFT
     *
     * @return int - returns 1 for success, 0 for failure
     *
     * */
    int fft2d(std::vector<std::complex<double> > *complex_image, int rows, int cols, int direction);

    /**
     * mat_to_array:
     * Takes an OpenCV Mat Object and converts into a array of vectors
     * std::vector obj []
     * */
    void real_to_complex_matrix(cv::Mat image, std::vector<std::complex<double> > *complex_image);

    /**
     * complex_matrix_to_mat:
     * Takes an array of complex<double> vectors, number of rows, and number of columns and
     * returns a cv::Mat representation of the 2 Dimensional matrix
     *
     * @param complex_image - the 2 Dimensional matrix to transform
     * @param rows - the number of rows in the matrix
     * @param cols - the number of columns in the matrix
     *
     * @return cv::Mat - the image representation of the 2 Dimensional matrix
     * */
    void complex_matrix_to_mat(std::vector<std::complex<double> > *complex_image, cv::Mat &image, int rows, int cols);

    /**
     * center_frequency_on_mat:
     * Centers quadrants of a resulting FFT transform around the middle of an image
     * This allows for a more understandable representation of the frequency domain image
     *
     * @param image - the image to center
     * */
    void center_frequency_on_mat(cv::Mat &image);


    /**
     * fft1d:
     * Takes a double array of real numbers and a double array of imaginary values.
     * Performs a 1 Dimensional FFT on the two sets.
     *
     * @param dir - the direction of the transform. 1: Forward, 2: Reverse
     * @param m -
     * @param x - the real values to transform
     * @param y - the imaginary values to transform
     *
     * @return:
     *      1 -
     *      0 -
     * */
    int fft1d(int dir, int m, double *x, double *y);



    /**
     * Loads an image from the given filename into the Mat at location
     *
     * @return:
     *      1 - success
     *      0 - failure
     * */
    int load_image(char * file_name, cv::Mat &location, int height, int width);

    /**
     * display_image:
     * Displays the given Mat in a window fram.
     * Exit the window by pressing any key.
     *
     * @param: image - the image to display
     * */
    void display_image(cv::Mat image);

    /**
     * save_image:
     * Saves the given Mat at the given file_name
     *
     * @param file_name - relative location of file to save to.
     * @param image - the Mat to save locally
     * */
    int save_image(char * file_name, cv::Mat image);

    /**
     * Converts an image to grayscale
     *
     * returns:
     *      1 - success
     *      0 - failure
     * */
    int rgb_to_gray(cv::Mat src, cv::Mat &dest);


private:
    int nearest_pow_2(int n, int *m, int *twopm);
    const WINDOW_SIZE_SM window_size_sm;
};