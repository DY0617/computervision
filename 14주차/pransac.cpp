#include "pransac.hpp"

using namespace panorama;

int LeastSquare::GenerateRandomNumber(int begin, int end)
{
	return std::uniform_int_distribution<int>(begin, end)(rnd);
}

void LeastSquare::DrawPoint(cv::InputOutputArray& first_canvas, cv::InputOutputArray& second_canvas)
{
	cv::Mat first_canvas_mat = first_canvas.getMat();
	cv::Mat second_canvas_mat = second_canvas.getMat();

	for (auto& point : distance_pair)
	{
		first_canvas_mat.at<uchar>(point.firstImgPtr.y, point.firstImgPtr.x) = 255;
		second_canvas_mat.at<uchar>(point.secondImgPtr.y, point.secondImgPtr.x) = 255;
	}
}

void LeastSquare::DrawLine(EquationElement element, cv::InputOutputArray& canvas)
{
	cv::Mat canvas_mat = canvas.getMat();
	for (int i = 0; i < canvas_mat.cols; i++) //직선의 방정식을 영상에 출력한다.
	{
		double sum3 = (element.m * i) + element.b;

		if (sum3 < 0)
			sum3 = 0;
		if (sum3 > canvas_mat.cols - 1)
			sum3 = canvas_mat.cols - 1;

		canvas_mat.at<uchar>(sum3, i) = 255;
	}
}

void PRANSAC::GetBestRotationMat(double minimum_error)
{
	double minimal_h_matrix_error = DBL_MAX;  // double 자료형의 최대값
	cv::Mat best_T;

	for (int i = 0; i < 1000; i++)
	{
		// [1] 3개의 서로다른 점들을 구한다.
		assert(distance_pair.size() >= 3);

		int a_idx, b_idx, c_idx;
		a_idx = b_idx = c_idx = -1;

		a_idx = GenerateRandomNumber(0, distance_pair.size() - 1);

		do
			b_idx = GenerateRandomNumber(0, distance_pair.size() - 1);
		while (a_idx == b_idx);

		do
			c_idx = GenerateRandomNumber(0, distance_pair.size() - 1);
		while (a_idx == c_idx || b_idx == c_idx);

		const PointSet a = distance_pair[a_idx];
		const PointSet b = distance_pair[b_idx];
		const PointSet c = distance_pair[c_idx];

		// [2] 세 점을 이용하여 변환행렬 homogeneous_matrix를 구한다.
		//     여기서 구한 homogeneous_matrix는 [알고리즘 7-9]의 T_j에 해당한다.
		std::vector<PointSet> selected_points{ a, b, c };

		cv::Mat homogeneous_matrix;
		GetHomogeneousMatrix(selected_points, homogeneous_matrix);

		// [3] 인라이어를 초기화한다.
		std::vector<PointSet> inliers{ a, b, c };

		// [4] T_j에 대한 오차를 모든 Distance Pair에 대해 계산하고,
		// minimum_error보다 작은 경우 해당 pair를 inlier에 넣는다.
		for (auto& pair : distance_pair)
		{
			if (pair == a || pair == b || pair == c)
				continue;

			if (GetError(homogeneous_matrix, pair) < minimum_error)
				inliers.push_back(pair);
		}

		// [5] Inlier가 많을수록 homogeneous_matrix가 강인함을 의미한다.
		//     이 inlier가 전체 Point의 70% 이상일 때, 이 inlier points 안에서
		//     다시 homogeneous_matrix를 계산한다.
		if (inliers.size() > distance_pair.size() * 0.7)
			GetHomogeneousMatrix(inliers, homogeneous_matrix);

		double sum_error = 0.0;
		for (auto& pair : distance_pair)
			sum_error += GetError(homogeneous_matrix, pair);

		// [6] 계산된 오차가 더 작은 경우,
		//     해당하는 오차값과 Homogeneous Matrix를 minimal_error_matrix로 설정한다.
		if (sum_error < minimal_h_matrix_error)
		{
			minimal_h_matrix_error = sum_error;
			best_T = homogeneous_matrix;
		}

		// [6] 오차가 제일 작은 Matrix를 Optimal Matrix(T)로 설정한다.
		T = cv::Mat::zeros(3, 3, CV_64FC1);

		T.at<double>(0, 0) = best_T.at<double>(0);	T.at<double>(0, 1) = best_T.at<double>(3);
		T.at<double>(1, 0) = best_T.at<double>(1);	T.at<double>(1, 1) = best_T.at<double>(4);
		T.at<double>(2, 0) = best_T.at<double>(2);	T.at<double>(2, 1) = best_T.at<double>(5);
		T.at<double>(2, 2) = 1;
	}
}

double PRANSAC::GetError(const cv::InputArray& homogeneous_matrix, const PointSet& pair)
{
	cv::Mat H_mat = homogeneous_matrix.getMat();

	cv::Mat A = cv::Mat::zeros(1, 3, CV_64FC1);
	cv::Mat T = cv::Mat::zeros(3, 3, CV_64FC1);

	// Homogeneous Matrix를 이용하여 Pair의 첫번째 원소를 Forward Transformation합니다.
	// 이 때, Matrix A의 값을 채우세요. "A.at<double>(0) = ?" 과 같은 방식으로 할당하세요.
	A.at<double>(0) = pair.firstImgPtr.y;
	A.at<double>(1) = pair.firstImgPtr.x;
	A.at<double>(2) = 1;

	// H_mat을 이용하여 행렬 T에 값을 할당하세요. (교재 p.322 수식 7.11 참조)
	T.at<double>(0, 0) = H_mat.at<double>(0, 0);	T.at<double>(0, 1) = H_mat.at<double>(3, 0);
	T.at<double>(1, 0) = H_mat.at<double>(1, 0);	T.at<double>(1, 1) = H_mat.at<double>(4, 0);
	T.at<double>(2, 0) = H_mat.at<double>(2, 0);	T.at<double>(2, 1) = H_mat.at<double>(5, 0);
	T.at<double>(2, 2) = 1;

	// A와 T를 이용하여 행렬 B를 구하세요.
	cv::Mat B = A * T;

	double error = pow((B.at<double>(0) - pair.secondImgPtr.y), 2) +
		pow((B.at<double>(1) - pair.secondImgPtr.x), 2);

	return error;
}

void PRANSAC::GetHomogeneousMatrix(const std::vector<PointSet>& selected_points, const cv::OutputArray& homogeneous_matrix)
{
	// Hint: A dot B = C
	cv::Mat A = cv::Mat::zeros(6, 6, CV_64FC1);
	cv::Mat C = cv::Mat::zeros(6, 1, CV_64FC1);

	// 교재 p.322의 수식 (7.14)에서의
	// 첫번째 6*6 배열을 생성합니다.
	for (auto& pair : selected_points)
	{
		int a_i1 = pair.firstImgPtr.y;
		int a_i2 = pair.firstImgPtr.x;

		// 교재나 강의자료를 참고하여 Matrix A를 채우시오.
		// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
		A.at<double>(0, 0) += a_i1 * a_i1;
		A.at<double>(1, 0) += a_i1 * a_i2;
		A.at<double>(2, 0) += a_i1;
		A.at<double>(0, 1) += a_i1 * a_i2;
		A.at<double>(1, 1) += a_i2 * a_i2;
		A.at<double>(2, 1) += a_i2;
		A.at<double>(0, 2) += a_i1;
		A.at<double>(1, 2) += a_i2;
		A.at<double>(2, 2) += 1;
		A.at<double>(3, 3) += a_i1 * a_i1;
		A.at<double>(4, 3) += a_i1 * a_i2;
		A.at<double>(5, 3) += a_i1;
		A.at<double>(3, 4) += a_i1 * a_i2;
		A.at<double>(4, 4) += a_i2 * a_i2;
		A.at<double>(5, 4) += a_i2;
		A.at<double>(3, 5) += a_i1;
		A.at<double>(4, 5) += a_i2;
		A.at<double>(5, 5) += 1;


		// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

	}

	// 교재 p.322의 수식 (7.14)에서의
	// 마지막의 결과 6*1 배열을 생성합니다.
	for (auto& pair : selected_points)
	{
		int a1 = pair.firstImgPtr.y;		int a2 = pair.firstImgPtr.x;
		int b1 = pair.secondImgPtr.y;		int b2 = pair.secondImgPtr.x;

		// 교재나 강의자료를 참고하여 Matrix C를 채우시오.
		// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
		C.at<double>(0,0) += a1 * b1;
		C.at<double>(1,0) += a2 * b1;
		C.at<double>(2,0) += b1;
		C.at<double>(3,0) += a1 * b2;
		C.at<double>(4,0) += a2 * b2;
		C.at<double>(5,0) += b2;
		// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
	}

	// 위에서 구한 행렬 A, C를 이용해서
	// 수식 (7.14) 중간의 [t_11 ~ t_32] 행렬 B을 구하세요.
	// ** 지금부터 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **
	cv::Mat B = A.inv()*C;

	// ** 여기까지 코드를 작성하세요. 이 줄은 지우시면 안 됩니다 **

	B.copyTo(homogeneous_matrix);
}

void PRANSAC::AttachImage(const cv::InputArray& first_image, const cv::InputArray& second_image, const cv::OutputArray& output_image)
{
	cv::Size first_image_size = first_image.size();
	output_image.create(cv::Size(first_image_size.width * 1.5, first_image_size.height), first_image.type());

	cv::Mat first_image_mat = first_image.getMat();
	cv::Mat second_image_mat = second_image.getMat();
	cv::Mat T_mat_inversed = T.inv();
	cv::Mat output_mat = output_image.getMat();

	cv::Mat homogeneous_vector(1, 3, CV_64FC1);

	// 첫번째 이미지를 결과 캔버스에 복사한다.
	for (int y = 0; y < first_image_mat.rows; y++)
		for (int x = 0; x < first_image_mat.cols; x++)
			output_mat.at<cv::Vec3b>(y, x) = first_image_mat.at<cv::Vec3b>(y, x);

	// 두번째 이미지의 값에 해당하는 좌표를 변환하여
	// 결과 캔버스의 변환좌표 위치에 두번째 이미지의 값을 복사한다.
	for (int y = 0; y < second_image_mat.rows; y++)
		for (int x = 0; x < second_image_mat.cols; x++)
		{
			homogeneous_vector.at<double>(0) = y;
			homogeneous_vector.at<double>(1) = x;
			homogeneous_vector.at<double>(2) = 1;

			cv::Mat homogeneous_result = homogeneous_vector * T_mat_inversed;

			int dy = homogeneous_result.at<double>(0);
			int dx = homogeneous_result.at<double>(1);

			if (dy < 0 || output_mat.rows <= dy || dx < 0 || output_mat.cols <= dx)
				continue;

			output_mat.at<cv::Vec3b>(dy, dx) = second_image_mat.at<cv::Vec3b>(y, x);
		}
}
