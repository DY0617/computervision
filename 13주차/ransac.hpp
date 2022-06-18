#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

namespace algorithms
{
	struct EquationElement
	{
		double m; // 기울기
		double b; // 절편
	};

	class LeastSquare
	{
	private:
		std::random_device rn;
		std::mt19937_64 rnd;

	public:
		std::vector<cv::Point> point_vector_list;  // 랜덤으로 생성된 점들을 저정한 벡터

		LeastSquare() : rn(), rnd(rn()), point_vector_list() { }
		int generate_random_number(int begin, int end);
		void fill_random_points(int inliers = 20, int outliers = 20);
		void draw_point(cv::InputOutputArray& canvas);
		EquationElement calculate_least_square();
		void draw_line(EquationElement element, cv::InputOutputArray& canvas);
	};

	class CRANSAC : public LeastSquare
	{
	private:
		EquationElement lineEquation;
		EquationElement optimalEquation;
		int pt1Idx, pt2Idx;

	public:
		cv::Point pt1, pt2;
		int maxInlier;

		CRANSAC() : lineEquation(), optimalEquation(), pt1Idx(0), pt2Idx(0), maxInlier(0) {
			memset(&lineEquation, 0, sizeof(EquationElement));
			memset(&optimalEquation, 0, sizeof(EquationElement));
		}
		void get_two_different_points();
		void convert_two_point_to_line();
		void calculate_inlier();
		void draw_optimal_line(cv::InputOutputArray& canvas);
	};
}