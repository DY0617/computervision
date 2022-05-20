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
				float dx = (float)(img.at<uchar>(y, x + 1) - img.at<uchar>(y, x - 1));
				float dy = (float)(img.at<uchar>(y + 1, x) - img.at<uchar>(y - 1, x));

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
				cv::cartToPolar(dx, dy, Mag, Angle, true);

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

void SIFT_Descriptor::MakeDescriptor(SIFT_c& sift)
{
	// 에지 히스토그램 개수
	const int SIFT_ORI_HIST_BINS = 8;
	for (int i = 0; i < sift.keyPointVec.size(); ++i)
	{
		keyPoint& key = sift.keyPointVec[i];
		/*
		컴퓨터비전 257p
		Hess가 구현한 방법에서는 siz를 3 x sigma로 설정하였다.
		*/
		int siz = 3 * key.sigma;


		// 36개로 양자화하여 10도씩 에지 히스토그램을 만들어기 때문에
		// 원래 에지를 계산하기 위해서 10을 곱해줌
		float orientation = key.dominantOrient * 10;

		cv::Point beginPoint = cv::Point(key.x - (2 * siz), key.y - (2 * siz));
		cv::Point endPoint = cv::Point(key.x + (2 * siz), key.y + (2 * siz));

		// Octave와 Sigma에 맞는 영상으로 처리
		int sigmaIdx = sift.GetSigmaIdx(key.sigma);
		const cv::Mat& img = sift.octaveVec[key.octave].gauImage[sigmaIdx];

		// 중점에서 회전하는 복합 행렬을 구하시오.
		/*
		CV_EXPORTS_W Mat getRotationMatrix2D( Point2f center, double angle, double scale );
		center : 중점 ? cv::Point로 넘김 ex) cv::Point(centerX, centerY)
		angle : 회전 방향
		scale : 스케일(실습에서는 1)
		*/
		cv::Mat m = cv::getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2), orientation, 1);

		for (int h = beginPoint.y; h < endPoint.y; h += siz)
		{
			for (int w = beginPoint.x; w < endPoint.x; w += siz)
			{
				std::vector<float> orient;
				std::vector<float> mag;
				for (int h2 = h; h2 < h + siz; h2++)
				{
					for (int w2 = w; w2 < w + siz; w2++)
					{
						cv::Mat orgIdx(cv::Size(1, 2), CV_32FC1);

						orgIdx.at<float>(0) = h2;
						orgIdx.at<float>(1) = w2;

						// 복합 동차 행렬 계산
						int rotationIdxY = int(orgIdx.at<float>(0) * m.at<double>(0) + orgIdx.at<float>(1) * m.at<double>(1) + m.at<double>(2));
						int rotationIdxX = int(orgIdx.at<float>(0) * m.at<double>(3) + orgIdx.at<float>(1) * m.at<double>(4) + m.at<double>(5));

						if (rotationIdxX > 0 && rotationIdxX < img.cols - 1
							& rotationIdxY > 0 && rotationIdxY < img.rows - 1)
						{
							// 픽셀별 dx와 dy를 계산하시오.(rotationIdxY와 rotationIdxX를 이용하세요.)
							float dx = (float)(img.at<uchar>(rotationIdxY, rotationIdxX + 1) - img.at<uchar>(rotationIdxY, rotationIdxX - 1));
							float dy = (float)(img.at<uchar>(rotationIdxY + 1, rotationIdxX) - img.at<uchar>(rotationIdxY - 1, rotationIdxX));

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
							cv::cartToPolar(dx, dy, Mag, Angle, true);

							orient.push_back(Angle.at<double>(0));
							mag.push_back(Mag.at<double>(0));
						}
					}
				}


				float hist[SIFT_ORI_HIST_BINS] = { 0, };
				for (int s = 0; s < orient.size(); ++s)
				{
					//bin = 8단계로 양자화된 에지 방향 값
					int bin = cvRound((SIFT_ORI_HIST_BINS / 360.f) * orient[s]);
					if (bin >= SIFT_ORI_HIST_BINS)
						bin -= SIFT_ORI_HIST_BINS;
					if (bin < 0)
						bin += SIFT_ORI_HIST_BINS;

					hist[bin] += (1 * mag[s]);

				}

				float sum_of_hist = 0;
				// 정규화
				for (int quant_radius = 0; quant_radius < SIFT_ORI_HIST_BINS; ++quant_radius)
				{
					sum_of_hist += hist[quant_radius];
				}

				for (int quant_radius = 0; quant_radius < SIFT_ORI_HIST_BINS; ++quant_radius)
				{
					if (sum_of_hist != 0)
						hist[quant_radius] /= sum_of_hist;
				}


				SIFT_Desc des(hist);
				SIFT_DescVec.push_back(des);
			}
		}
	}
}