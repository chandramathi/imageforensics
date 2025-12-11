#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;
/**
 * @brief 
 * The core function that performs contrast adaptive hough transform
 * to segment the pupil mask from the eye segment
 * pupil being the darkest region in the eye
 * @param eyeGray The input grayscale image patch containing the isolated eye region
 * @param pupilMask Output parameter: The binary mask generated for the detected pupil.
 * @param center Output parameter: The coordinates ($\text{Point}$) of the detected pupil center.
 * @param radius Output parameter: The radius ($\text{int}$) of the detected pupil.
 * @param cannyLow Parameter: The lower threshold value used for the Canny edge detection pre-processing step.
 * @param cannyHigh Parameter: The upper threshold value used for the Canny edge detection pre-processing step.
 * @param houghMinR Parameter: The minimum radius to search for in the Hough transform algorithm.
 * @param houghMaxR Parameter: The maximum radius to search for in the Hough transform algorithm.
 * @param dp Parameter: The inverse ratio of the accumulator resolution to the image resolution (specific to OpenCV's HoughCircles).
 * @param minDist Parameter: The minimum distance required between the centers of detected circles.
 * @param houghParam1 
 * @param houghParam2 
 * @return the status as sucess or failure of segmentation 
 */
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
