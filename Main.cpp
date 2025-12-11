#include <iostream>
#include <opencv2/opencv.hpp>
#include "FaceSegmentation.h"
#include "EyeSegmentation.h"
#include "BIoU.h"
#include "PupilSegment.h"

using namespace std;
using namespace cv;
/**
 * @brief 
 * Checks if the provided string path or identifier represents a video file or an image
 * @param p The input string of the file path.
 * @return true 
 * @return false 
 */
bool isVideo(const string& p)
{
    string lower = p;
    for (auto& c : lower) c = tolower(c);

    return (lower.size() >= 4 &&
           (lower.substr(lower.size()-4)==".mp4" ||
            lower.substr(lower.size()-4)==".avi" ||
            lower.substr(lower.size()-4)==".mov"));
}
/**
 * @brief 
 * Displays the detected eye region, superimposes the pupil mask, and prints the BIoU score for visual verification.
 * @param eye The input image patch containing the isolated eye region.
 * @param maskGray The grayscale image mask representing the detected pupil
 * @param biou The Bounding Box IoU score calculated for the detected pupil.
 */
void showEyeAndMask(const Mat& eye, const Mat& maskGray, double biou)
{
    Mat maskResized, maskColor, combined;

    if (maskGray.size() != eye.size()) {
        resize(maskGray, maskResized, eye.size(), 0, 0, INTER_NEAREST);
    } else {
        maskResized = maskGray.clone();
    }

    cvtColor(maskResized, maskColor, COLOR_GRAY2BGR);

    std::string label = "BIoU = " + std::to_string(biou).substr(0, 6);
    int font = FONT_HERSHEY_SIMPLEX;
    double fs = 0.6;
    int th = 2;

    putText(maskColor, label, Point(10, 25),
                font, fs, Scalar(0, 255, 0), th);

    hconcat(eye, maskColor, combined);

    imshow("Eye + Mask", combined);
    waitKey(0);
}
/**
 * @brief 
 * 
 * @param input 
 * @param display 
 */
void runEyeMode(const string& input, bool display)
{
    Mat eye = imread(input);
    if (eye.empty()) {
        cerr << "Could not read input image.\n";
        return;
    }

    Mat norm = normalizeEyeCrop(eye);

    Mat gray; cvtColor(norm, gray, COLOR_BGR2GRAY);

    Mat mask;
    Point center; int radius;

    if (!findPupilMask(gray, mask, center, radius)) {
        cerr << "Pupil not found.\n";
        return;
    }

    // find contour
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        cerr << "No contour found.\n";
        return;
    }
    double biou = computeBIoU(mask, contours[0]);
    cout << "BIoU = " << biou << endl;

    if (display) {
        showEyeAndMask(norm, mask, biou);
    }
}
/**
 * @brief Processes an image input stream specifically in face detection when
 * face parameter is used in the commandline argument and extracts eye and pupil from it
 * @param input Path to the image file or camera device index to be processed.
 * @param display Boolean flag to indicate whether the processing results should be displayed in a window.
 */
void runFaceMode(const string& input, bool display)
{
    Mat left, right;
    std::vector<Point> leftPts, rightPts;
    if (!extractEyesFromFace(input, left, right,leftPts,rightPts)) {
        cerr << "Face/eye extraction failed.\n";
        return;
    }

    // LEFT EYE
    Mat grayL; cvtColor(left, grayL, COLOR_BGR2GRAY);
    Mat maskL;
    Point centerL; int radiusL;

    if (!findPupilMask(grayL, maskL, centerL, radiusL)) {
        cerr << "Left pupil not found.\n";
        return;
    }

    vector<vector<Point>> contoursL;
    findContours(maskL, contoursL, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contoursL.empty()) {
        cerr << "No contour for left eye.\n";
        return;
    }

    double biouL = computeBIoU(maskL, contoursL[0]);
    cout << "Left Eye BIoU = " << biouL << endl;

    // RIGHT EYE
    Mat grayR; cvtColor(right, grayR, COLOR_BGR2GRAY);
    Mat maskR;
    Point centerR; int radiusR;

    if (!findPupilMask(grayR, maskR, centerR, radiusR)) {
        cerr << "Right pupil not found.\n";
        return;
    }

    vector<vector<Point>> contoursR;
    findContours(maskR, contoursR, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contoursR.empty()) {
        cerr << "No contour for right eye.\n";
        return;
    }

    double biouR = computeBIoU(maskR, contoursR[0]);
    cout << "Right Eye BIoU = " << biouR << endl;

    if (display) {
        showEyeAndMask(left, maskL, biouL);
        showEyeAndMask(right, maskR, biouR);
    }
}
/**
 * @brief 
 * Executes the processing pipeline on a video file, applying analysis frame-by-frame.
 * @param input Path to the video file or the index of the camera device to be used and only .mp4 files
 * @param maxFrames Maximum number of frames to process before stopping (use 0 or a negative value to process the entire video)
 * @param display Boolean flag to control whether the video output and analysis results should be displayed in real-time.
 */
void runVideoMode(const string& input, int maxFrames, bool display)
{
    VideoCapture cap(input);
    if (!cap.isOpened()) {
        cerr << "Cannot open video.\n";
        return;
    }

    cout << "Processing first " << maxFrames << " frames\n";

    Mat frame;
    for (int i = 0; i < maxFrames; i++) {

        if (!cap.read(frame)) break;

        string tempName = "__temp_frame.jpg";
        imwrite(tempName, frame);

        Mat left, right;
        std::vector<Point> leftPts, rightPts;
        if (!extractEyesFromFace(tempName, left, right,leftPts,rightPts)) {
            cout << "Frame " << i << ": No face detected\n";
            continue;
        }

        // LEFT EYE
        Mat gL; cvtColor(left, gL, COLOR_BGR2GRAY);
        Mat mL;
        Point cL; int rL;
        if (findPupilMask(gL, mL, cL, rL)) {
            vector<vector<Point>> cnt;
            findContours(mL, cnt, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            if (!cnt.empty()) {
                double biouL = computeBIoU(mL, cnt[0]);
                cout << "Frame " << i << " - Left Eye BIoU = " << biouL << endl;

                if (display) {
                    showEyeAndMask(left, mL, biouL);
                }
            }
        }

        // RIGHT EYE
        Mat gR; cvtColor(right, gR, COLOR_BGR2GRAY);
        Mat mR;
        Point cR; int rR;
        if (findPupilMask(gR, mR, cR, rR)) {
            vector<vector<Point>> cnt2;
            findContours(mR, cnt2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            if (!cnt2.empty()) {
                double biouR = computeBIoU(mR, cnt2[0]);
                cout << "Frame " << i << " - Right Eye BIoU = " << biouR << endl;

                if (display) {
                    showEyeAndMask(right, mR, biouR);
                }
            }
        }
    }
}

/**
 * @brief 
 * The core function that runs the command line tool checkPupil which can take one file at a time
 * and process pupil based generated image classification
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char** argv)
{
    if (argc < 2) {
        cerr << "Usage: ./checkPupil --eye=\"input_eye.jpg\" | --face=\"input_face.jpg\" | --video=\"input.mp4\" [--display on/off] [--frames numFrames]\n";
        return 1;
    }

    string input;
    string mode;
    bool display = true;
    int numFrames = 30;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg.rfind("--eye=", 0) == 0) {
            mode = "eye";
            input = arg.substr(6); 
        }
        else if (arg.rfind("--face=", 0) == 0) {
            mode = "face";
            input = arg.substr(7); 
        }
        else if (arg.rfind("--video=", 0) == 0) {
            mode = "video";
            input = arg.substr(8);
        }
        else if (arg == "--display" && i + 1 < argc) {
            display = (string(argv[++i]) == "on");
        }
        else if (arg == "--frames" && i + 1 < argc) {
            numFrames = stoi(argv[++i]);
        }
    }

    if (mode.empty() || input.empty()) {
        cerr << "No input file specified.\n";
        return 1;
    }

    if (mode == "eye") {
        runEyeMode(input, display);
    }
    else if (mode == "face") {
        runFaceMode(input, display);
    }
    else if (mode == "video") {
        runVideoMode(input, numFrames, display);
    }
    else {
        cerr << "Invalid mode.\n";
        return 1;
    }

    return 0;
}
