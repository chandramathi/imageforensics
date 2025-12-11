#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 @brief Extracts the left and right eye regions from a detected face image.
  dlib library to extract the left and right eyes Which are further send down for pupil extraction
 This function performs facial landmark detection, identifies the eye landmarks,
 and then crops and possibly normalizes the image patches corresponding to the eyes.
 @param imagePath Path to the input image file containing a face.
 @param leftEye Output parameter: The extracted image patch containing the left eye.
 @param rightEye Output parameter: The extracted image patch containing the right eye.
 @param leftLandmarks Output parameter: A vector of specific landmark points (e.g., dlib indices) found for the left eye.
 @param rightLandmarks Output parameter: A vector of specific landmark points (e.g., dlib indices) found for the right eye.
 @return bool Returns true if the eyes were successfully extracted and landmarks were found, false otherwise.
 */
bool extractEyesFromFace(const string& path, Mat& left, Mat& right, vector<Point>& leftLandmarks, vector<Point>& rightLandmarks);
