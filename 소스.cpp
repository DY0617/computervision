#include "opencv2/opencv.hpp"

using namespace cv;

int main() {

	cv::Mat image = cv::imread("lenna.bmp", cv::IMREAD_COLOR);

	cv::imwrite("lenna_converted.jpg", image);

	return 0;
}