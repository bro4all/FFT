#include "gtest/gtest.h"
#include "fft.h"

class fftFixture : public ::testing::Test {

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {


    }


public:
    cv::Mat emptyMat;
};

TEST_F(fftFixture, basicTest) {
    fft fft;
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

TEST_F(fftFixture, loadImageTest){
    fft fft;
    cv::Mat image;
    EXPECT_EQ(1, fft.load_image("../../img.JPG", image));
}

TEST_F(fftFixture, displaImageTest){
    fft fft;
    cv::Mat image;
    EXPECT_EQ(1, fft.load_image("../../img.JPG", image));
    fft.display_image(image);
}

TEST_F(fftFixture, convertGray){
    fft fft;
    cv::Mat image;
    EXPECT_EQ(1, fft.load_image("../../img.JPG", image));

    // test valid image given and display the gray result
    EXPECT_EQ(1, fft.rgb_to_gray(image, image));
    EXPECT_NO_THROW(fft.display_image(image));

    // test empty image given
    EXPECT_EQ(0, fft.rgb_to_gray(emptyMat, emptyMat));
}