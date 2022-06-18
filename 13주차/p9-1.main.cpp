/*
 * 과제10: RANSAC
 * -> "algorithms/ransac.hpp" 참조
 */

#include "ransac.hpp"
#include <iostream>

using namespace algorithms;

int main0()
{
	cv::Mat canvas(cv::Size(280, 280), CV_8UC1, cv::Scalar(0));

	LeastSquare least_square;
	EquationElement element;

	least_square.fill_random_points();
	least_square.draw_point(canvas);
	element = least_square.calculate_least_square();

	least_square.draw_line(element, canvas);

	cv::imshow("Least Square Result", canvas);
	cv::imwrite("leastsquare.bmp", canvas);
	cv::waitKey(0);

	return 0;
}