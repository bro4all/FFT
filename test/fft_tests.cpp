#include "gtest/gtest.h"
#include "fft.h"

class fftFixture : public ::testing::Test {

protected:
    virtual void TearDown() {
        remove(output_file_name);
    }

    virtual void SetUp() {
        
    }


public:
    cv::Mat emptyMat;
    char * output_file_name = (char*)("../../Gray_img.JPG");
    char * box = (char*)("../../square1.jpg");
    char * boat = (char*)("../../img.JPG");
    char * clown = (char*)("../../cln1.jpg");
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
    EXPECT_EQ(1, fft.load_image(boat, image, 256, 256));
}

TEST_F(fftFixture, displayImageTest){
    fft fft;
    cv::Mat image;
    EXPECT_EQ(1, fft.load_image(boat, image, 256, 256));
    EXPECT_NO_THROW(fft.display_image(image));
}

TEST_F(fftFixture, saveImageTest){
    fft fft;
    cv::Mat image;

    // Load a given image
    fft.load_image(boat, image, 256, 256);

    // Save image, 1 for success
    EXPECT_EQ(1, fft.save_image(output_file_name, image));

    // Load the image we just saved, expect 1 for success
    EXPECT_EQ(1, fft.load_image(output_file_name, image, 256, 256));
}

TEST_F(fftFixture, convertGray){
    fft fft;
    cv::Mat image;
    EXPECT_EQ(1, fft.load_image(boat, image, 256, 256));

    // test valid image given and display the gray result
    EXPECT_EQ(1, fft.rgb_to_gray(image, image));
    EXPECT_NO_THROW(fft.display_image(image));

    // test empty image given
    EXPECT_EQ(0, fft.rgb_to_gray(emptyMat, emptyMat));
}

TEST_F(fftFixture, fftForwardTest){
    fft fft;
    cv::Mat image;
    EXPECT_EQ(1, fft.load_image(boat, image, 256, 256));

    // test valid image given and display the gray result
    EXPECT_EQ(1, fft.rgb_to_gray(image, image));


    std::vector<std::complex<double>>  x[256];
    fft.real_to_complex_matrix(image, x);
    fft.fft2d(x, 256, 256, 1);

    // to reverse uncomment this line
    //fft.fft2d(x, 256, 256, -1);

    cv::Mat fourier_image;
    fft.complex_matrix_to_mat(x, fourier_image, 256, 256);
    //fft.center_frequency_on_mat(fourier_image);
    fft.display_image(fourier_image);
}