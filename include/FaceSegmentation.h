#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

bool extractEyesFromFace(const string& path, Mat& left, Mat& right, vector<Point>& leftLandmarks, vector<Point>& rightLandmarks);
