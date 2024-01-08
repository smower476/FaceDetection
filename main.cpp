//#include "opencv2/opencv.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
 
using namespace std;
using namespace cv;
 
int main(int argc, char **argv){
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap(argv[1]); 
    CascadeClassifier facedetect;
    facedetect.load("/home/chicken/Downloads/FaceDetection/haarcascade_frontalface_default.xml");
    // Check if camera opened successfully
    if(!cap.isOpened()){
      cout << "Error opening video stream or file" << endl;
      return -1;
    }
     
    while(1){ 
        Mat frame;
        // Capture frame-by-frame
        cap >> frame;
      
        // If the frame is empty, break immediately
        if (frame.empty())
            break;
        
        vector<Rect> faces; 
       
        facedetect.detectMultiScale(frame, faces, 1.3, 5);
        
        for(int i = 0; i < faces.size(); i++){
            rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(50, 50, 255), 3);
        } 



        // Display the resulting frame
        imshow( "Frame", frame );
     
        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
          break;
    }
  
    // When everything done, release the video capture object
    cap.release();
 
    // Closes all the frames
    destroyAllWindows();
     
    return 0;
}

