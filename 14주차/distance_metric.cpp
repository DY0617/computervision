#include "distance_metric.hpp"
#include <vector>

using namespace panorama;
using namespace std;
using namespace cv;

void DistanceCalculator::ExtractMinDistance(const vector<HOGPair>& h1, const vector<HOGPair>& h2)
{
	vector<PointSet> vector;
	double threshold = 900;

	for (auto pair1 : h1)
	{
		double min_distance = DBL_MAX;
		cv::Point pair2_min_dist_pt;

		for (auto pair2 : h2)
		{
			double distance = CalculateDistance(pair1.hog, pair2.hog);

			if (min_distance > distance)
			{
				min_distance = distance;
				pair2_min_dist_pt = pair2.keypoint;
			}
		}

		if (threshold > min_distance)
		{
			PointSet data { pair1.keypoint, pair2_min_dist_pt, min_distance };
			vector.push_back(data);
		}
	}

	distance_pair.assign(vector.begin(), vector.end());
}

double DistanceCalculator::CalculateDistance(const vector<double>& h1, const vector<double>& h2)
{

	double result = 0;
	for (int i = 0; i < h1.size(); i++)
	{
		double hist1_bin_value = h1.at(i);
		double hist2_bin_value = h2.at(i);

		// 두 Histogram Bin 값 간의 Euclidean Distance를 구하세요.
		// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
		result += pow(hist1_bin_value - hist2_bin_value, 2);
		if (i == h1.size()-1) {
			result = sqrt(result);
		}
		// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
	}

	return result;
}

void DistanceCalculator::DisplayPairLine(const Mat& image1, const Mat& image2)
{

	Mat result;
	hconcat(image1, image2, result);

	cvtColor(result, result, cv::COLOR_GRAY2RGB);

	for (auto& pair : distance_pair)
	{
		cv::Point img1_px = pair.firstImgPtr;
		cv::Point img2_px = pair.secondImgPtr;

		line(result, Point(img1_px.x, img1_px.y), Point(img2_px.x + image1.cols, img2_px.y), Scalar(0, 0, 255));
	}

	imshow("Mathcing Points", result);
	waitKey();
}