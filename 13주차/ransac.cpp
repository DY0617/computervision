#include "ransac.hpp"

using namespace algorithms;

int algorithms::LeastSquare::generate_random_number(int begin, int end)
{
	return std::uniform_int_distribution<int>(begin, end)(rnd);
}

void algorithms::LeastSquare::fill_random_points(int inliers, int outliers)
{
	// inlier
	for (int i = 0; i < inliers; i++)
	{
		int x = generate_random_number(0, 279);
		int y = generate_random_number(120, 125);

		point_vector_list.emplace_back(x, y);
	}

	// outlier
	for (int i = 0; i < outliers; i++)
	{
		int x = generate_random_number(0, 279);
		int y = generate_random_number(0, 279);

		point_vector_list.emplace_back(x, y);
	}
}

void algorithms::LeastSquare::draw_point(cv::InputOutputArray& canvas)
{
	cv::Mat canvas_mat = canvas.getMat();
	for (auto& point : point_vector_list)
		canvas_mat.at<uchar>(point.y, point.x) = 255;
}

EquationElement algorithms::LeastSquare::calculate_least_square()
{
	double sumX = 0.0;
	double sumY = 0.0;

	double avgX = 0.0;
	double avgY = 0.0;

	// 기울기와 절편을 구하세요.
	for (auto& point : point_vector_list)
	{
		// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
		sumX += point.x;
		sumY += point.y;
		// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
	}

	avgX = sumX / point_vector_list.size();
	avgY = sumY / point_vector_list.size();

	double sumNumerator = 0.0;
	double sumDenominator = 0.0;

	for (auto& point : point_vector_list)
	{
		// Numerator와 Denominator는 각각 분자와 분모를 의미함
		// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
		sumNumerator += (point.x - avgX) * (point.y - avgY);
		sumDenominator += ((double)point.x - avgX) * ((double)point.x - avgX);
		// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
	}

	EquationElement elem;
	// Numerator(분자)와 Denominator(분모)를 구한 값으로 기울기와 절편을 구하세요.
	// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
	elem.m = sumNumerator / sumDenominator;
	elem.b = avgY - elem.m * avgX;
	// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

	return elem;
}

void algorithms::LeastSquare::draw_line(EquationElement element, cv::InputOutputArray& canvas)
{
	cv::Mat canvas_mat = canvas.getMat();
	for (int i = 0; i < canvas_mat.cols; i++) //직선의 방정식을 영상에 출력
	{
		double sum3 = (element.m * i) + element.b;

		if (sum3 < 0)
			sum3 = 0;
		if (sum3 > canvas_mat.cols - 1)
			sum3 = canvas_mat.cols - 1;

		canvas_mat.at<uchar>(sum3, i) = 255;
	}
}

void CRANSAC::get_two_different_points()
{
	pt1Idx = generate_random_number(0, point_vector_list.size() - 1);
	pt1 = point_vector_list[pt1Idx];

	pt2Idx = generate_random_number(0, point_vector_list.size() - 1);
	pt2 = point_vector_list[pt2Idx];

	// 같은 점인 경우 다시 고른다.
	while (pt1 == pt2)
	{
		pt2Idx = generate_random_number(0, point_vector_list.size() - 1);
		pt2 = point_vector_list[pt2Idx];
	}
}

void CRANSAC::convert_two_point_to_line()
{
	// Point pt1과 Point pt2를 이용해서 기울기와 절편을 계산하세요
	double avY = (pt1.y + pt2.y) / 2;
	double avX = (pt1.x + pt2.x) / 2;
	lineEquation.m = ((pt1.y - avY) * (pt1.x - avX) + (pt2.y - avY)
		* (pt2.x - avX)) / ((pt1.x - avX) * (pt1.x - avX)
			+ (pt2.x - avX) * (pt2.x - avX));
	lineEquation.b = avY - lineEquation.m * avX;
}

void CRANSAC::calculate_inlier()
{
	std::vector<cv::Point> liner;
	for (int i = 0; i < point_vector_list.size(); i++)
	{
		if (i != pt1Idx && i != pt2Idx)
		{
			// distance = |(-(x * m) - b + y) / root(m^2 + 1)|
			// point_vector_list[i]를 이용해서 거리를 계산하세요
			// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
			double d= (-(point_vector_list[i].x * lineEquation.m)
				- lineEquation.b + point_vector_list[i].y) /
				(double)sqrt(pow(lineEquation.m, 2) + 1);
			if (d < 0)
				d = -d;

			double distance = d;
			// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
			if (distance <= 5)
				liner.push_back(point_vector_list[i]);
		}
	}
	if (maxInlier < liner.size())
	{
		maxInlier = liner.size();
		optimalEquation.m = lineEquation.m;
		optimalEquation.b = lineEquation.b;
	}
}

void algorithms::CRANSAC::draw_optimal_line(cv::InputOutputArray& canvas)
{
	draw_line(optimalEquation, canvas);
}
