#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;
// Finds pupil mask and returns center/radius.
// - eyeGray: single-channel eye image (grayscale).
// - pupilMask: output binary mask (CV_8UC1) same size as eyeGray (255 inside pupil).
// - center: pupil center (x,y).
// - radius: pupil radius (pixels).
// Returns true on success (found a pupil), false otherwise.
bool findPupilMask(const cv::Mat &eyeGray,
                   cv::Mat &pupilMask,
                   cv::Point &center,
                   int &radius,
                   // optional tuning parameters:
                   int cannyLow = 30,
                   int cannyHigh = 90,
                   int houghMinR = 10,
                   int houghMaxR = 120,
                   double dp = 1.2,
                   int minDist = 30,
                   int houghParam1 = 80,
                   int houghParam2 = 30);
