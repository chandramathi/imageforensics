#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include "FaceSegmentation.h"

using namespace cv;
using namespace std;

/** This function adds a few pixels while segmenting a bit more than the original
returned through dlib because when the eyelids are narrow or almost closed the
further operations down the pipeline will throw an error because of lack of pixels to work with
 @param shape dlib facial landmark detection result (contains coordinates for all facial features).
 @param idx List of landmark indices corresponding specifically to the points of the eye.
 @param w Desired padding width to add to the eye box.
 @param h Desired padding height to add to the eye box.
*/
static dlib::rectangle expandEyeBox( const dlib::full_object_detection& shape, const std::vector<int>& idx, int w, int h)
{
    long minx = shape.part(idx[0]).x();
    long maxx = minx;
    long miny = shape.part(idx[0]).y();
    long maxy = miny;

    for (int i : idx) {
        minx = std::min(minx, shape.part(i).x());
        maxx = std::max(maxx, shape.part(i).x());
        miny = std::min(miny, shape.part(i).y());
        maxy = std::max(maxy, shape.part(i).y());
    }

    long margin = 30;
    minx -= margin;  miny -= margin;
    maxx += margin;  maxy += margin;

    long width = maxx - minx;
    long height = maxy - miny;
    long side = max(width, height);

    long cx = (minx + maxx) / 2;
    long cy = (miny + maxy) / 2;

    long half = side / 2;
    // long x1 = max<long>(0, cx - half);
    long x1 = std::max<long>(0L, (long)(cx - half));
    long y1 = std::max<long>(0, cy - half);
    long x2 = std::min<long>(w - 1, cx + half);
    long y2 = std::min<long>(h - 1, cy + half);

    return dlib::rectangle(x1, y1, x2, y2);
}

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
bool extractEyesFromFace(const string& imagePath, Mat& leftEye, Mat& rightEye, std::vector<Point>& leftLandmarks, std::vector<Point>& rightLandmarks)
{
    dlib::array2d<dlib::rgb_pixel> img;
    try { load_image(img, imagePath); }
    catch (...) { 
        cout<<"Image loading failed";
        return false; 
    }

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor sp;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

    auto dets = detector(img);
    if (dets.empty()) return false;

    dlib::full_object_detection shape = sp(img, dets[0]);

    vector<int> leftIdx  = {36,37,38,39,40,41};
    vector<int> rightIdx = {42,43,44,45,46,47};

    int W = img.nc(), H = img.nr();

    auto Lrect = expandEyeBox(shape, leftIdx,  W, H);
    auto Rrect = expandEyeBox(shape, rightIdx, W, H);

    dlib::array2d<dlib::rgb_pixel> Limg, Rimg;
    assign_image(Limg, sub_image(img, Lrect));
    assign_image(Rimg, sub_image(img, Rrect));

    leftEye  = Mat(Limg.nr(), Limg.nc(), CV_8UC3);
    rightEye = Mat(Rimg.nr(), Rimg.nc(), CV_8UC3);

    for (long r = 0; r < Limg.nr(); r++)
        for (long c = 0; c < Limg.nc(); c++)
            leftEye.at<Vec3b>(r,c) =
                Vec3b(Limg[r][c].blue, Limg[r][c].green, Limg[r][c].red);

    for (long r = 0; r < Rimg.nr(); r++)
        for (long c = 0; c < Rimg.nc(); c++)
            rightEye.at<Vec3b>(r,c) =
                Vec3b(Rimg[r][c].blue, Rimg[r][c].green, Rimg[r][c].red);

    leftLandmarks.clear();
    rightLandmarks.clear();

    for (int idx : leftIdx) {
        Point p(shape.part(idx).x(), shape.part(idx).y());

        int x = p.x - Lrect.left();
        int y = p.y - Lrect.top();

        if (x >= 0 && y >= 0 &&
            x < leftEye.cols && y < leftEye.rows)
        {
            leftLandmarks.emplace_back(x, y);
        }
    }

    for (int idx : rightIdx) {
        Point p(shape.part(idx).x(), shape.part(idx).y());

        int x = p.x - Rrect.left();
        int y = p.y - Rrect.top();

        if (x >= 0 && y >= 0 &&
            x < rightEye.cols && y < rightEye.rows)
        {
            rightLandmarks.emplace_back(x, y);
        }
    }

    return true;
}
