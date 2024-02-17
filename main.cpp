#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include "cheeks.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	//Gpu Info
	cuda::printCudaDeviceInfo(0);

	cout << "Enter the name of the mask file || press ENTER if you dont want to use it" << endl;
	string mask;
	getline(cin, mask);


	
	VideoCapture cap(argv[1]);
	if (!cap.isOpened()) {
		cerr << "Error opening video stream or file" << endl;
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

		if (mask == "") {
			CheeksDetection(faces, img);
		} else {
			CheeksDetection(faces, img, mask);
		}

		auto end = getTickCount();
		auto totalTime = (end - start) / getTickFrequency();
		auto fps = 1 / totalTime;
		putText(img, "FPS: " + to_string(int(fps)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);
		imshow("Mask", img);
		if (waitKey(1) == 'q') {
			break;
		}

	}

	cap.release();
	destroyAllWindows();
	return 0;
}