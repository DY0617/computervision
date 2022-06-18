#include "SIFT.h"
#include "SIFT_Descriptor.h"
using namespace cv;
void main()
{
	Mat src;
	/// Load image
	src = imread("Cliff.bmp", cv::IMREAD_GRAYSCALE);	
	Mat dst;
	cvtColor(src, dst, cv::COLOR_GRAY2BGR);

	SIFT_c sift;
	cv::GaussianBlur(src, src, cv::Size(0, 0), 1.6, 1.6);

	sift.InitSig();

	sift.MakeOctaveImg(src);
	sift.DOG();
	sift.FindKeyPoint();
	
	for(int i = 0; i < 3; ++i)
	{
		int octIdx = sift.octaveVec.size()-1;
		Octave &oct = sift.octaveVec[octIdx];

		cv::Mat resizeMat;
		resize(oct.gauImage[3], resizeMat, cv::Size(oct.gauImage[3].cols/2, oct.gauImage[3].rows/2));
		sift.MakeOctaveImg(resizeMat);
		sift.DOG();
		sift.FindKeyPoint();
	}
	
	sift.DrawKeyPoint(dst);
	imshow("key Point", dst);

	waitKey();
	//SIFT_Descriptor descriptor;
	//descriptor.FindDominantOrientation(sift);
	/*descriptor.MakeDescriptor(sift);

	// 첫번째 기술자만 프린트함
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			for (int z = 0; z < 8; ++z)
			{
				std::cout << descriptor.SIFT_DescVec[i * 4 + j].discriptor[z] << " ";
			}
			std::cout << std::endl;
		}
	}

	cv::waitKey();*/
}
