/*
 * 과제10: RANSAC
 * -> "algorithms/sift.hpp" 참조
 */

#include "ransac.hpp"
#include <iostream>

using namespace algorithms;

int main()
{
	cv::Mat canvas(cv::Size(280, 280), CV_8UC1, cv::Scalar(0));

	CRANSAC ransac;
	EquationElement element;

	ransac.fill_random_points(80, 80);
	ransac.draw_point(canvas);

	for (int l = 0; l < 2000; l++)
	{
		ransac.get_two_different_points();
		ransac.convert_two_point_to_line();
		ransac.calculate_inlier();
	}

	ransac.draw_optimal_line(canvas);

	cv::imshow("RANSAC Result", canvas);
	cv::imwrite("ransac.bmp", canvas);
	cv::waitKey(0);

	return 0;
}