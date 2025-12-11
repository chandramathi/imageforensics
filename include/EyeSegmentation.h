#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief 
 * Preprocessing step to normalize an input segmented Eye
 * @param eye matrix of pixels
 * @return  normalized eye matrix
 */
cv::Mat normalizeEyeCrop(const cv::Mat& eye);
