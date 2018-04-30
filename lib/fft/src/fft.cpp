#include "fft.h"

fft::fft() {}


fft::~fft() {

}

/**
 * Cooley Tukey Algo - Wikipedia
 * Unoptimizeded
  X0,...,N−1 ← ditfft2(x, N, s):             DFT of (x0, xs, x2s, ..., x(N-1)s):
    if N = 1 then
        X0 ← x0                                      trivial size-1 DFT base case
    else
        X0,...,N/2−1 ← ditfft2(x, N/2, 2s)             DFT of (x0, x2s, x4s, ...)
        XN/2,...,N−1 ← ditfft2(x+s, N/2, 2s)           DFT of (xs, xs+2s, xs+4s, ...)
        for k = 0 to N/2−1                           combine DFTs of two halves into full DFT:
            t ← Xk
            Xk ← t + exp(−2πi k/N) Xk+N/2
            Xk+N/2 ← t − exp(−2πi k/N) Xk+N/2
        endfor
    endif
*/
void fft::fast_fourier_transform(std::vector<std::complex<double>> &x)
{
    const int N = x.size();
    if (N <= 1) return;

    // Divide the sequence into even and odd

    // Even
    std::vector<std::complex<double>> even;
    for(int i = 0; i < x.size(); i += 2) {
        even.push_back(x[i]);
    }

    // Odd
    std::vector<std::complex<double>> odd;
    for(int i = 1; i < x.size(); i += 2) {
        odd.push_back(x[i]);
    }


    // Perform FFT on both
    fast_fourier_transform(even);
    fast_fourier_transform(odd);

    // Recombine even and odd
    for (int k = 0; k < N/2; ++k)
    {
        std::complex<double> t = std::polar(1.0, -2 * PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k+N/2] = even[k] - t;
    }
}

// Inverse FFT
void fft::inverse_fft(std::vector<std::complex<double>> &x)
{
    for(int i = 0; i < x.size(); i ++) {
        x[i] = std::conj(x[i]);
    }
    // Perform FFT, FFT is the inverse of itself.
    fast_fourier_transform(x);

    for(int i = 0; i < x.size(); i ++) {
        x[i] = std::conj(x[i]);
    }

    // scale the numbers
    for(int i = 0; i < x.size(); i ++) {
        x[i] = x[i]/double(x.size());
    }
}