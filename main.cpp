#include "opencv2/opencv.hpp"
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
    facedetect.load("haarcascades/haarcascade_frontalface_default.xml");

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

        //Cheeks Detection
        Mat gray_image;
        cvtColor(frame, gray_image, COLOR_BGR2GRAY);

        facedetect.detectMultiScale(gray_image, faces, 1.3, 5);

    // Iterate through detected faces and highlight cheeks
    for (const auto& face : faces) {
        // Extract the region of interest (ROI) for the face
        Mat face_roi = gray_image(face);

        // Define the region of interest for the left cheek (adjust these values based on your needs)
        int x_cheekA = face.x + face.width / 6;
        int y_cheekA = face.y + face.height / 2;
        int width_cheekA = face.width / 4;
        int height_cheekA = face.height / 4;

        // Define the region of interest for the right cheek (adjust these values based on your needs)
        int x_cheekB = face.x + face.width * 4 / 6;
        int y_cheekB = face.y + face.height / 2;
        int width_cheekB = face.width / 4;
        int height_cheekB = face.height / 4;       

        // Draw a rectangle around the cheeks
        rectangle(frame, Rect(x_cheekA, y_cheekA, width_cheekA, height_cheekA), Scalar(0, 255, 0), 2);
        rectangle(frame, Rect(x_cheekB, y_cheekB, width_cheekB, height_cheekB), Scalar(0, 255, 0), 2);
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

