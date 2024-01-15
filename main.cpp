#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

//		vector<cv::Rect> faces;
int CheeksDetection(vector<cv::Rect>& tmpRect, Mat& tmpImg) {
	Mat overlayImage = imread("overlay_image.png", IMREAD_UNCHANGED);
	
	if (overlayImage.empty()) {
		std::cerr << "Error loading overlayImage." << std::endl;
		return -1;
	}
	


	for (const auto& face : tmpRect) {
		int x_cheekA = face.x + face.width / 6;
		int y_cheekA = face.y + face.height / 2;
		int width_cheekA = face.width / 4;
		int height_cheekA = face.height / 4;

		int x_cheekB = face.x + face.width * 4 / 6;
		int y_cheekB = face.y + face.height / 2;
		int width_cheekB = face.width / 4;
		int height_cheekB = face.height / 4;

		cv::Rect roiRectA(x_cheekA, y_cheekA, width_cheekA, height_cheekA);
		cv::Rect roiRectB(x_cheekB, y_cheekB, width_cheekB, height_cheekB);
		cv::Mat roi = tmpImg(roiRectA).clone();

		Mat resized_image;
		resize(overlayImage, resized_image, Size(width_cheekA, height_cheekA), INTER_LINEAR);

		
		//cv::medianBlur(roi, roi, 111);

		if (resized_image.channels() != roi.channels()) {
			cv::cvtColor(resized_image, resized_image, cv::COLOR_RGBA2RGB);
		}

		//cout << "ROIrows " << roi.rows << " ROIcols " << roi.cols << " channels " << roi.channels() << endl;
		//cout << "rows " << resized_image.rows << " cols " << resized_image.cols << " channels " << resized_image.channels() << endl;
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
					//tmpImg(roiRect).setTo(cv::Scalar(0, 0, 255));

				}
			}
		}



	

		

		// Draw a rectangle around the cheeks
		rectangle(tmpImg, Rect(x_cheekA, y_cheekA, width_cheekA, height_cheekA), Scalar(0, 255, 0), 2);
		rectangle(tmpImg, Rect(x_cheekB, y_cheekB, width_cheekB, height_cheekB), Scalar(0, 255, 0), 2);
	}
	return 1;
}


int main(int argc, char** argv) {
	//Gpu Info
	cuda::printCudaDeviceInfo(0);

	VideoCapture cap(argv[1]);
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	cv::Ptr<cv::cuda::CascadeClassifier> faceCascade = cv::cuda::CascadeClassifier::create("haarcascades/haarcascade_frontalface_alt.xml");
	Mat img;
	cuda::GpuMat imgGpu;	
	

	while (cap.isOpened()) {
		auto start = getTickCount();

		cap.read(img);
		if (img.empty())
			break;
		std::vector<cv::Rect> faces;

		imgGpu.upload(img);
		cuda::cvtColor(imgGpu, imgGpu, COLOR_BGR2GRAY);

		cv::cuda::GpuMat faces_cuda;

		faceCascade->detectMultiScale(imgGpu, faces_cuda);

		//imgGpu.download(img);
		faceCascade->convert(faces_cuda, faces);


		for (int i = 0; i < faces.size(); i++) {
			rectangle(img, faces[i].tl(), faces[i].br(), Scalar(50, 50, 255), 3);
		}
		CheeksDetection(faces ,img);
		auto end = getTickCount();
		auto totalTime = (end - start) / getTickFrequency();
		auto fps = 1 / totalTime;
		putText(img, "FPS: " + to_string(int(fps)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);
		imshow("Image", img);
		if (waitKey(1) == 'q') {
			break;
		}

	}

	cap.release();
	_sleep(10000);
	destroyAllWindows();
	return 0;
}