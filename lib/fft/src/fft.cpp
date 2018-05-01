#include "fft.h"

fft::fft() {

}


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


    // Perform fft1d on both
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

// Inverse fft1d
void fft::inverse_fft(std::vector<std::complex<double>> &x)
{
    for(int i = 0; i < x.size(); i ++) {
        x[i] = std::conj(x[i]);
    }
    // Perform FFT, fft1d is the inverse of itself.
    fast_fourier_transform(x);

    for(int i = 0; i < x.size(); i ++) {
        x[i] = std::conj(x[i]);
    }

    // scale the numbers
    for(int i = 0; i < x.size(); i ++) {
        x[i] = x[i]/double(x.size());
    }
}

int fft::load_image(char * file_name, cv::Mat &location, int height, int width){
    location = cv::imread(file_name, CV_LOAD_IMAGE_COLOR);
    cv::resize(location, location, cv::Size(height, width), 0, 0, cv::INTER_CUBIC);

    if(!location.data) {
        std::cout << "Image not found!" << std::endl;
        return 0;//not found
    }else{
        return 1;//found
    }
}

void fft::display_image(cv::Mat image){
    cv::namedWindow("Display window", CV_WINDOW_NORMAL);
    cv::resizeWindow("Display window", 100, 100);
    cv::imshow( "Display window", image);
    cv::waitKey(0);
}

int fft::save_image(char * file_name, cv::Mat image){
    if(!image.empty()) {
        cv::imwrite(file_name, image);
        std::cout << "New image saved." << std::endl;
        return 1;
    }else{
        std::cout << "New image couldn't be saved." << std::endl;
        return 0;
    }

}

/**
 * rgb_to_gray:
 * Takes a Mat source and a Mat address destination.
 * Converts a BGR 2 Gray scale image.
 * */
int fft::rgb_to_gray(cv::Mat src, cv::Mat &dest) {
    if(src.empty())
        return 0;

    cv::cvtColor(src, dest, CV_BGR2GRAY);
    return 1;
}

/**
 * real_to_complex_matrix:
 * Takes a Mat image and transforms it into an array of complex vectors giving an output of a
 * complex 2 dimensional matrix.
 *
 * The complex matrix contains the real values from the Mat and 0's for imaginary.
 * complex<double> = (image.real(), 0)
 *
 * @param image - the OpenCV Mat containing the image.
 * @param complex_image - vector of complex doubles to store the image data.
 * */
void fft::real_to_complex_matrix(cv::Mat image, std::vector<std::complex<double>> *complex_image) {
    // Temporary storage for our double values.
    std::vector<double> temp[image.rows];

    // Copy the Mat data into a vector<double>
    for(int i = 0; i < image.rows; i ++) {
        image.row(i).copyTo(temp[i]);
    }

    // Convert the vector<double> into complex<double> and store in output
    for (int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            std::complex<double> val;
            val.real(temp[i][j]);
            val.imag(0);
            complex_image[i].push_back(val);
        }
    }
}


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
int fft::fft2d(std::vector<std::complex<double>> *complex_image, int rows, int cols, int direction) {
    int m,twopm;
    double *real,*imag;

    // Transform each row
    real = new double[cols];
    imag = new double[cols];
    if (!nearest_pow_2(cols, &m, &twopm) || twopm != cols)
        return 0;

    for (int j=0;j<rows;j++) {
        for (int i=0;i<cols;i++) {
            real[i] = complex_image[i][j].real();
            imag[i] = complex_image[i][j].imag();
        }
        fft1d(direction, m, real, imag);
        for (int i=0;i<cols;i++) {
            complex_image[i][j].real(real[i]);
            complex_image[i][j].imag(imag[i]);
        }
    }
    delete real;
    delete imag;


    // Transform each column
    real = new double[rows];
    imag = new double[rows];
    if (!nearest_pow_2(rows, &m, &twopm) || twopm != rows)
        return 0;

    for (int i=0;i<cols;i++) {
        for (int j=0;j<rows;j++) {
            real[j] = complex_image[i][j].real();
            imag[j] = complex_image[i][j].imag();
        }
        fft1d(direction, m, real, imag);
        for (int j=0;j<rows;j++) {
            complex_image[i][j].real(real[j]);
            complex_image[i][j].imag(imag[j]);
        }
    }
    delete real;
    delete imag;
}

int fft::fft1d(int dir, int m, double *x, double *y) {
    long nn=0,i1=0,k=0,i2=0,l1 =0,l2 = 0;
    double c1=0,c2=0,tx=0,ty=0,t1=0,t2=0,u1=0,u2=0,z=0;

    /* Calculate the number of points */
    nn = 1;
    for (int i=0;i<m;i++)
        nn *= 2;

    /* Do the bit reversal */
    i2 = nn >> 1;
    int j = 0;
    for (int i=0;i<nn-1;i++) {
        if (i < j) {
            tx = x[i];
            ty = y[i];
            x[i] = x[j];
            y[i] = y[j];
            x[j] = tx;
            y[j] = ty;
        }
        k = i2;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
    /* Compute the fft1d */
    c1 = -1.0;
    c2 = 0.0;
    l2 = 1;
    for (int l=0;l<m;l++) {
        l1 = l2;
        l2 <<= 1;
        u1 = 1.0;
        u2 = 0.0;
        for (int j=0;j<l1;j++) {
            for (int i=j;i<nn;i+=l2) {
                i1 = i + l1;
                t1 = u1 * x[i1] - u2 * y[i1];
                t2 = u1 * y[i1] + u2 * x[i1];
                x[i1] = x[i] - t1;
                y[i1] = y[i] - t2;
                x[i] += t1;
                y[i] += t2;
            }
            z =  u1 * c1 - u2 * c2;
            u2 = u1 * c2 + u2 * c1;
            u1 = z;
        }
        c2 = sqrt((1.0 - c1) / 2.0);
        if (dir == 1)
            c2 = -c2;
        c1 = sqrt((1.0 + c1) / 2.0);
    }

    // Scale values for reverse transform
    if (dir == -1) {
        for (int i=0;i<nn;i++) {
            x[i] /= (double)nn;
            y[i] /= (double)nn;
        }
    }

    return(true);

}

int fft::nearest_pow_2(int n, int *m, int *twopm)
{
    if (n <= 1) {
        *m = 0;
        *twopm = 1;
        return(false);
    }

    *m = 1;
    *twopm = 2;
    do {
        (*m)++;
        (*twopm) *= 2;
    } while (2*(*twopm) <= n);

    if (*twopm != n)
        return(false);
    else
        return(true);
}

void fft::complex_matrix_to_mat(std::vector<std::complex<double> > complex_image[], cv::Mat &image, int rows, int cols) {
    uchar data[rows][cols];
    for(int i = 0; i < rows; i ++) {
        for(int j = 0; j < cols; j++) {
            data[i][j] = sqrt( pow( complex_image[i][j].real(), 2 ) + pow( complex_image[i][j].imag(),2 ) );
            //data[i][j] = 255*(log(1 +  data[i][j]));
        }
    }
    /**
     * Return Mat object
     * Calling clone() is needed to create memory for the uchar[][] data above which is
     * deleted at the end of this functions scope.
     * */
    image = cv::Mat(rows, cols, CV_8UC1, data).clone();
}

void fft::center_frequency_on_mat(cv::Mat &image) {
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = image.cols/2;
    int cy = image.rows/2;

    cv::Mat q0(image, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(image, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(image, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(image, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    cv::normalize(image, image, 0, 255, CV_MINMAX);
}