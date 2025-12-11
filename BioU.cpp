#include "BIoU.h"
#include "Ellipse.h"
using namespace cv;

double computeBIoU(const Mat& mask, const std::vector<Point>& contour)
{
    if (contour.size() < 5) return 0.0;
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
