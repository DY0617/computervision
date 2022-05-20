#pragma once
#include "opencv2\opencv.hpp"
#include <vector>

class keyPoint
{
public:
	keyPoint(int y, int x, int oct, double sig) : y(y), x(x), octave(oct), sigma(sig){}
	int y;
	int x;
	int octave;
	double sigma;
	float dominantOrient;
};

class Octave
{
public:
	cv::Mat gauImage[6];
	cv::Mat DoG[5];
	int octave;
};

class SIFT_c
{
public:
	std::vector<Octave> octaveVec;
	std::vector<keyPoint> keyPointVec;
	double sig[6];
public:
	void InitSig();
	void MakeOctaveImg(cv::Mat& org);
	void DOG();
	void FindKeyPoint();
	void DrawKeyPoint(cv::Mat &dst);
	int GetSigmaIdx(float sigma)
	{
		for(int i = 0; i < 6; ++i)
			if(sigma == sig[i])
				return i;
	}
};