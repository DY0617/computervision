#include "opencv2/opencv.hpp"

using namespace cv;

int main() {
	cv::Mat image;
	image = cv::imread("lenna.bmp", cv::IMREAD_GRAYSCALE);

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (image.at < uchar>(y, x) < 128)
				image.at < uchar>(y, x) = 0;
			else
				image.at<uchar>(y, x) = 255;
		}
	}

	cv::imwrite("lenna_binarized1.jpg", image);

	return 0;
}