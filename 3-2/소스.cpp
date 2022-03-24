#include "opencv2/opencv.hpp"

using namespace cv;


int main() {

	cv::Mat image, result_image;
	image = cv::imread("lenna.bmp", cv::IMREAD_COLOR);
	image.copyTo(result_image);



	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			int b = image.at<Vec3b>(y, x)[0];
			int g = image.at<Vec3b>(y, x)[1];
			int r = image.at<Vec3b>(y, x)[2];


			result_image.at<Vec3b>(y, x)[0] = 0.299 * r + 0.587 * g + 0.114 * b;
			result_image.at<Vec3b>(y, x)[1] = 0.5 * r - 0.419 * g - 0.081 * b + 128;
			result_image.at<Vec3b>(y, x)[2] = -0.169 * r - 0.331 * g + 0.5 * b + 128;
			//y=0.299 * r + 0.587 * g + 0.114 * b
			//cr=-0.169 * r - 0.331 * g + 0.5 * b + 128
			//cb=0.5 * r - 0.419 * g - 0.081 * b + 128
		}
	}

	cv::imshow("original", image);
	cv::imshow("Result", result_image);
	cv::waitKey(0);
	return 0;
}

