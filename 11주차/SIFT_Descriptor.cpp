#include "SIFT_Descriptor.h"

void SIFT_Descriptor::FindDominantOrientation(SIFT_c& sift)
{
	// 에지 히스토그램 개수
	const int SIFT_ORI_HIST_BINS = 36;
	for (int i = 0; i < sift.keyPointVec.size(); i++)
	{
		keyPoint& key = sift.keyPointVec[i];

		/*
		컴퓨터비전 256p
		Hess가 구현한 방법에서는 1.5 x 3 x Sigma를 반올림하여 w를 구한 후
		(2w+1) x (2w+1)의 윈도우를 사용하였다.
		*/
		int radius = 1.5 * 3 * key.sigma;


		std::vector<float> orient;
		std::vector<float> mag;

		// Octave와 Sigma에 맞는 영상으로 처리
		int sigmaIdx = sift.GetSigmaIdx(key.sigma);
		const cv::Mat& img = sift.octaveVec[key.octave].gauImage[sigmaIdx];
		for (int h = -radius; h <= radius; ++h)
		{
			int y = key.y + h;
			if (y <= 0 || y >= img.rows - 1)
				continue;
			for (int w = -radius; w <= radius; w++)
			{
				int x = key.x + w;
				if (x <= 0 || x >= img.cols - 1)
					continue;

				// 픽셀별 dx와 dy를 계산하시오.
				// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
				float dx = img.at<float>(y, x - 1) * (-1) + img.at<float>(y, x + 1);
				float dy = img.at<float>(y - 1, x) * (-1) + img.at<float>(y + 1, x);
				// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

				cv::Mat Mag(cv::Size(1, 1), CV_32FC1);
				cv::Mat Angle(cv::Size(1, 1), CV_32FC1);

				/*
					CV_EXPORTS_W void cartToPolar(InputArray x, InputArray y,
							  OutputArray magnitude, OutputArray angle,
							  bool angleInDegrees = false)
					x : dx
					y : dy
					magnitude : 에지 크기
					angle : 에지 방향
				*/
				// 에지 방향과 에지 크기를 계산하시오.
				
				// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
				Angle.at<double>(0) = atan(dy / dx) + 90;
				Mag.at<double>(0) = sqrt(pow(dx, 2) + pow(dy, 2));
				// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

				orient.push_back(Angle.at<double>(0));
				mag.push_back(Mag.at<double>(0));
			}
		}

		float hist[SIFT_ORI_HIST_BINS] = { 0, };
		for (int s = 0; s < orient.size(); ++s)
		{
			//bin = 10도 간격으로 양자화된 에지 방향 값
			int bin = cvRound((SIFT_ORI_HIST_BINS / 360.f) * orient[s]);
			if (bin >= SIFT_ORI_HIST_BINS)
				bin -= SIFT_ORI_HIST_BINS;
			if (bin < 0)
				bin += SIFT_ORI_HIST_BINS;
			// 히스토그램의 값을 에지의 방향과 강도의 곱으로 계산
			hist[bin] += (1 * mag[s]);
		}

		// 
		float maxval = hist[0];
		float maxOri = 0;
		// 값이 최대인 히스토그램 값을 Dominant Orienation으로 설정
		for (int ii = 1; ii < SIFT_ORI_HIST_BINS; ii++)
		{
			if (maxval < hist[ii])
			{
				maxval = hist[ii];
				maxOri = ii;
			}
		}

		key.dominantOrient = maxOri;
	}


}

