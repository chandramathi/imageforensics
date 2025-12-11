#include <opencv2/opencv.hpp>
 
/**
 * @brief 
 * Computes the Bounding Box Intersection over Union metric between a detected circular mask and the ground truth eye contour landmarks.
 * @param mask The binary image mask representing the detected pupil region.
 * @param contour A vector of points defining the ground truth contour
 * @return the value of the BIou Score 
 */
double computeBIoU(const cv::Mat& mask, const std::vector<cv::Point>& contour);
