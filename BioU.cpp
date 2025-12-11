#include "BIoU.h"
#include "Ellipse.h"
using namespace cv;
/**
 * @brief 
 * Computes the Bounding Box Intersection over Union metric between a detected circular mask and the ground truth eye contour landmarks.
 * @param mask The binary image mask representing the detected pupil region.
 * @param contour A vector of points defining the ground truth contour
 * @return the value of the BIou Score 
 */
double computeBIoU(const Mat& mask, const std::vector<Point>& contour)
{
    if (contour.size() < 5) return 0.0;
    //Computes ellipse fitting and the BIoU score based on the custom ellipse fitting implemented in
    //this project
    try{
        CustomEllipseFitter fitter;
        RotatedRect ellipseBox = fitter.fit(contour);
        if (ellipseBox.size.width <= 0.0 || ellipseBox.size.height <= 0.0) {
            std::cerr << "Warning: Custom ellipse fit produced invalid geometry (non-positive axes)." << std::endl;
            if (std::isnan(ellipseBox.size.width) || std::isnan(ellipseBox.size.height)) {
                std::cerr << "Severe Warning: Ellipse axes are NaN (numerical error)." << std::endl;
            }
        }
        Mat ellipseMask = Mat::zeros(mask.size(), CV_8UC1);
        ellipse(ellipseMask, ellipseBox, Scalar(255), -1, LINE_AA);

        Mat inter, uni;
        bitwise_and(mask, ellipseMask, inter);
        bitwise_or(mask, ellipseMask, uni);

        double I = countNonZero(inter);
        double U = countNonZero(uni);

        return (U == 0.0 ? 0.0 : I / U);
    }catch(...){
        //If the custom ellipse fitting failed due to any errors then the OpenCV's own
        //fit ellipse function is used as a fallback. 
        RotatedRect ellipseBox = fitEllipse(contour);
        if (ellipseBox.size.width <= 0.0 || ellipseBox.size.height <= 0.0) {
            std::cerr << "Warning: Custom ellipse fit produced invalid geometry (non-positive axes)." << std::endl;
            if (std::isnan(ellipseBox.size.width) || std::isnan(ellipseBox.size.height)) {
                std::cerr << "Severe Warning: Ellipse axes are NaN (numerical error)." << std::endl;
            }
        }
        Mat ellipseMask = Mat::zeros(mask.size(), CV_8UC1);
        ellipse(ellipseMask, ellipseBox, Scalar(255), -1, LINE_AA);

        Mat inter, uni;
        bitwise_and(mask, ellipseMask, inter);
        bitwise_or(mask, ellipseMask, uni);

        double I = countNonZero(inter);
        double U = countNonZero(uni);

        return (U == 0.0 ? 0.0 : I / U);
    }
}
