#include <iostream>
#include <opencv2/opencv.hpp>
#include "cheeks.hpp"

using namespace std;
using namespace cv;

int CheeksDetection(vector<cv::Rect>& tmpRect, Mat& tmpImg) {
	for (const auto& face : tmpRect) {
		int x_cheekA = face.x + face.width / 6;
		int y_cheekA = face.y + face.height / 2;
		int width_cheekA = face.width / 4;
		int height_cheekA = face.height / 4;

		int x_cheekB = face.x + face.width * 4 / 6;
		int y_cheekB = face.y + face.height / 2;
		int width_cheekB = face.width / 4;
		int height_cheekB = face.height / 4;

		rectangle(tmpImg, Rect(x_cheekA, y_cheekA, width_cheekA, height_cheekA), Scalar(0, 255, 0), 2);
		rectangle(tmpImg, Rect(x_cheekB, y_cheekB, width_cheekB, height_cheekB), Scalar(0, 255, 0), 2);
	}
	return 0;
}

int CheeksDetection(vector<cv::Rect>& tmpRect, Mat& tmpImg, string& tmpmask) {
	Mat overlayImage = imread(tmpmask, IMREAD_UNCHANGED);
	if (overlayImage.empty()) {
		std::cerr << "Error loading overlayImage." << std::endl;
		return -1;
	}
	Mat resized_image;


	for (const auto& face : tmpRect) {
		int x_cheekA = face.x + face.width / 6;
		int y_cheekA = face.y + face.height / 2;
		int width_cheekA = face.width / 4;
		int height_cheekA = face.height / 4;

		int x_cheekB = face.x + face.width * 4 / 6;
		int y_cheekB = face.y + face.height / 2;
		int width_cheekB = face.width / 4;
		int height_cheekB = face.height / 4;

		rectangle(tmpImg, Rect(x_cheekA, y_cheekA, width_cheekA, height_cheekA), Scalar(0, 255, 0), 2);
		rectangle(tmpImg, Rect(x_cheekB, y_cheekB, width_cheekB, height_cheekB), Scalar(0, 255, 0), 2);




		cv::Rect roiRectA(x_cheekA, y_cheekA, width_cheekA, height_cheekA);
		cv::Rect roiRectB(x_cheekB, y_cheekB, width_cheekB, height_cheekB);
		cv::Mat roi = tmpImg(roiRectA).clone();

		resize(overlayImage, resized_image, Size(width_cheekA, height_cheekA), INTER_LINEAR);

		/////////////////////////
		if (resized_image.channels() != roi.channels()) {
			cv::cvtColor(resized_image, resized_image, cv::COLOR_RGBA2RGB);
		}

		if (resized_image.channels() != roi.channels() || roi.cols != resized_image.cols || resized_image.channels() != roi.channels()) {
			std::cerr << "Error: Number of channels mismatch between source and destination." << std::endl;
			return -1;
		}

		for (int i = 0; i < roi.cols; i++)
		{
			for (int j = 0; j < roi.rows; j++)
			{
				if (!(resized_image.at<Vec3b>(j, i) == Vec3b(255, 255, 255)))
				{
					tmpImg(roiRectA).at<Vec3b>(j, i) = Vec3b(255, 41, 251);// resized_image.at<Vec3b>(j, i);
					tmpImg(roiRectB).at<Vec3b>(j, i) = Vec3b(255, 41, 251);
				}
			}
		}
	}
	return 0;
}