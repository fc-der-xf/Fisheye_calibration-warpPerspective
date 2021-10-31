#include "Fisheye_calibration.h"

using namespace cv;
using namespace std;

int Fisheye_calibration(string source) {
	//读取图片
	cv::Mat src_mat = cv::imread(source);
	/*cv::imshow("test",src_mat);*/ //测试成功
	Size board_size = Size(4, 4); //棋盘格中内角点的个数。棋盘格有m * n个正方形，内角点就有m-1 * n-1个
	std::vector<Point2f> corners; //保存图中所有角点的坐标
	vector<vector<Point2f>>  corners_Seq; // 保存所有图像的角点
	cv::Mat gray_mat;
	cv::cvtColor(src_mat, gray_mat, CV_RGB2GRAY);
	

	bool isfind = cv::findChessboardCorners(src_mat, board_size, corners, CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
	if (isfind != true) {
		std::cout << "Can't find corners!Please check your image or board size." << std::endl;
	}
	else {
		//亚像素精确化
		cornerSubPix(gray_mat, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		corners_Seq.push_back(corners);
	}
	//进行标定
	Size square_size = Size(20, 20);
	vector<vector<Point3f>> object_Points; // 保存三维坐标点
	vector<Point3f> tempPointSet;
	for (int i = 0; i < board_size.height; i++)
	{
		for (int j = 0; j < board_size.width; j++)
		{
			/* 假设定标板放在世界坐标系中z=0的平面上 */
			Point3f tempPoint;
			tempPoint.x = i * square_size.width;
			tempPoint.y = j * square_size.height;
			tempPoint.z = 0;
			tempPointSet.push_back(tempPoint);
		}
	}
	object_Points.push_back(tempPointSet);
	Size image_size = src_mat.size();
	cv::Matx33d intrinsic_matrix; // 内参矩阵
	cv::Vec4d distortion_coeffs; // 畸变系数
	std::vector<cv::Vec3d> rotation_vectors;  // 旋转向量
	std::vector<cv::Vec3d> translation_vectors;  // 平移向量
	int flags = 0;
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
	fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
	std::cout << "test" << std::endl;
	
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "矫正图像" << endl;
	Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
	Mat t = src_mat.clone();
	cv::remap(src_mat, t, mapx, mapy, cv::INTER_LINEAR);

	//进行透视变换
	Point2f src_point[4];
	Point2f dst_point[4];
	src_point[0] = corners[0];
	src_point[1] = corners[3];
	src_point[2] = corners[12];
	src_point[3] = corners[15];

	dst_point[0] = Point(560.0,560.0);
	dst_point[1] = Point(720.0, 560.0);
	dst_point[2] = Point(560.0, 720.0);
	dst_point[3] = Point(720.0, 720.0);

	cv::Mat trans_mat;
	trans_mat = getPerspectiveTransform(src_point, dst_point);
	warpPerspective(t, t, trans_mat, t.size(), INTER_LINEAR, BORDER_CONSTANT);

	cv::waitKey(0);
	return 0;
}