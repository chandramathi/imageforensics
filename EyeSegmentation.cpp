#include "EyeSegmentation.h"
using namespace cv;

cv::Mat normalizeEyeCrop(const cv::Mat& eye)
{
    int w = eye.cols, h = eye.rows;
    int side = std::max(w, h);

    int top = (side - h) / 2;
    int bottom = side - h - top;
    int left = (side - w) / 2;
    int right = side - w - left;

    Mat padded;
    copyMakeBorder(eye, padded, top, bottom, left, right, BORDER_CONSTANT, Scalar(0,0,0));

    return padded;
}
