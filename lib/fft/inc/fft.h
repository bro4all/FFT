#include <math.h>
#include <complex>
#include <vector>
#include <iostream>

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

private:
};