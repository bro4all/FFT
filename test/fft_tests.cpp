#include "gtest/gtest.h"
#include "fft.h"

class fftFixture : public ::testing::Test {

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {


    }


public:

};

TEST_F(fftFixture, basicTest) {
    fft::fft fft;
    std::vector<std::complex<double>> data = { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };

    // forward fast_fourier_transform
    fft.fast_fourier_transform(data);

    std::cout << "\n\nIn the Fourier Domain: " << std::endl;
    for (int i = 0; i < 8; ++i)
    {
        std::cout << data[i] << std::endl;
    }

    // inverse fast_fourier_transform
    fft.inverse_fft(data);

    std::cout << std::endl << "inverse_fft" << std::endl;
    for (int i = 0; i < 8; ++i)
    {
        std::cout << data[i] << std::endl;
    }
}