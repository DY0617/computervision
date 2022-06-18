#include "SIFT.h"
#include "SIFT_Descriptor.h"
using namespace cv;
using namespace std;

int main()
{
	Mat image1;
	Mat image2;

	/// Load image
	image1 = imread("landscape-left.bmp", cv::IMREAD_GRAYSCALE);
	image2 = imread("landscape-right.bmp", cv::IMREAD_GRAYSCALE);

	Mat dst;
	Mat dst2;
	cvtColor(image1, dst, cv::COLOR_GRAY2BGR);
	cvtColor(image2, dst2, cv::COLOR_GRAY2BGR);

	SIFT_c sift;
	cv::GaussianBlur(image1, image1, cv::Size(0, 0), 1.6, 1.6);

	sift.InitSig();
	sift.MakeOctaveImg(image1);
	sift.DOG();
	sift.FindKeyPoint();

	SIFT_c sift2;
	cv::GaussianBlur(image2, image2, cv::Size(0, 0), 1.6, 1.6);

	sift2.InitSig();
	sift2.MakeOctaveImg(image2);
	sift2.DOG();
	sift2.FindKeyPoint();


	for (int i = 0; i < 3; ++i)
	{
		int octIdx = sift.octaveVec.size() - 1;
		Octave& oct = sift.octaveVec[octIdx];

		cv::Mat resizeMat;
		resize(oct.gauImage[3], resizeMat, cv::Size(oct.gauImage[3].cols / 2, oct.gauImage[3].rows / 2));
		sift.MakeOctaveImg(resizeMat);
		sift.DOG();
		sift.FindKeyPoint();
	}

	for (int i = 0; i < 3; ++i)
	{
		int octIdx = sift2.octaveVec.size() - 1;
		Octave& oct2 = sift2.octaveVec[octIdx];

		cv::Mat resizeMat;
		resize(oct2.gauImage[3], resizeMat, cv::Size(oct2.gauImage[3].cols / 2, oct2.gauImage[3].rows / 2));
		sift2.MakeOctaveImg(resizeMat);
		sift2.DOG();
		sift2.FindKeyPoint();
	}

	SIFT_Descriptor descriptor;
	descriptor.FindDominantOrientation(sift);
	descriptor.MakeDescriptor(sift);

	SIFT_Descriptor descriptor2;
	descriptor2.FindDominantOrientation(sift2);
	descriptor2.MakeDescriptor(sift2);


	cout << "(sift.keyPointVec.size())" << (sift.keyPointVec.size()) << endl;;

	//유클리디안 거리
	double sum = 0.0;
	double distance = 0.0;
	double min = INT16_MAX;
	int KeyPoint1_minIdx = 0;
	int KeyPoint2_minIdx = 0;



	for (int k = 0; k < sift.keyPointVec.size(); k++) {
		for (int l = 0; l <  sift2.keyPointVec.size(); l++) {
			for (int i = 0; i < 4; ++i)
			{
				for (int j = 0; j < 4; ++j)
				{
					for (int z = 0; z < 8; ++z)
					{

						sum += pow(descriptor.SIFT_DescVec[k*16+i*4+j].discriptor[z] - descriptor2.SIFT_DescVec[l*16+i*4+j].discriptor[z], 2);

					}
				}
			}
			distance = sqrt(sum);
			sum = 0.0;
			//std::cout <<"distance: " << distance <<  std::endl;
			//std::cout << std::endl;

			if (distance < 1) {
				min = distance;
				KeyPoint1_minIdx = k;
				KeyPoint2_minIdx = l;

				std::cout << "distance: " << min << std::endl;
				std::cout << "KeyPoint1_minIdx: " << KeyPoint1_minIdx << std::endl;
				std::cout << "KeyPoint2_minIdx: " << KeyPoint2_minIdx << std::endl;
				std::cout << std::endl;

				int scale = pow(2, sift.keyPointVec[k].octave);
				circle(dst, cv::Point(sift.keyPointVec[k].x * scale, sift.keyPointVec[k].y) * scale, 3, cv::Scalar(0, 0, 255));

				int scale2 = pow(2, sift2.keyPointVec[l].octave);
				circle(dst2, cv::Point(sift2.keyPointVec[l].x * scale2, sift2.keyPointVec[l].y) * scale2, 3, cv::Scalar(0, 0, 255));
				
			}
		}
	}
	imshow("key Point1", dst);
	imshow("key Point2", dst2);

	imwrite("ketPoint1.jpg", dst);
	imwrite("ketPoint2.jpg", dst2);

	cv::waitKey();

	return 0;
}