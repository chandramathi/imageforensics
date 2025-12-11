#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iomanip>
#include "FaceSegmentation.h"
#include "EyeSegmentation.h"
#include "BIoU.h"
#include "PupilSegment.h"

using namespace std;
using namespace cv;
namespace fs = std::__fs::filesystem;

bool isImageFile(const string& p)
{
    string s = p;
    for (auto& c : s) c = tolower(c);
    return (s.find(".jpg") != string::npos ||
            s.find(".png") != string::npos ||
            s.find(".jpeg") != string::npos);
}

bool isVideoFile(const string& p)
{
    string s = p;
    for (auto& c : s) c = tolower(c);
    return (s.find(".mp4") != string::npos ||
            s.find(".avi") != string::npos ||
            s.find(".mov") != string::npos);
}

void saveResultImage(const Mat& eye,
                     const Mat& mask,
                     const string& outPath,
                     double biou)
{
    Mat maskR, maskColor, combined;

    if (mask.size() != eye.size())
        resize(mask, maskR, eye.size(), 0, 0, INTER_NEAREST);
    else
        maskR = mask.clone();

    cvtColor(maskR, maskColor, COLOR_GRAY2BGR);

    string label = "BIoU = " + to_string(biou).substr(0,6);
    putText(maskColor, label, Point(10,25),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,0), 2);

    hconcat(eye, maskColor, combined);

    imwrite(outPath, combined);
}

void saveEyeAndMaskSeparately(const Mat& eye,
                              const Mat& mask,
                              const string& outDir,
                              const string& baseName)
{
    string eyePath  = outDir + "/" + baseName + "_eye.jpg";
    string maskPath = outDir + "/" + baseName + "_mask.jpg";

    imwrite(eyePath, eye);
    imwrite(maskPath, mask);
}

void drawLandmarks(Mat& img,
                   const std::vector<cv::Point>& pts)
{
    for (const auto& p : pts)
        circle(img, p, 2, Scalar(0,255,0), FILLED);
}

void saveFaceAnnotatedResult(const cv::Mat& eye,
                             const cv::Mat& mask,
                             const std::vector<cv::Point>& landmarks,
                             const std::string& outPath,
                             double biou)
{
    cv::Mat annotated = eye.clone();

    for (const auto& p : landmarks)
        cv::circle(annotated, p, 2, cv::Scalar(0,255,0), cv::FILLED);

    std::string label = "BIoU = " + std::to_string(biou).substr(0,6);
    cv::putText(annotated, label, cv::Point(10,25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0,255,0), 2);


    cv::Mat maskResized, maskColor;

    if (mask.size() != annotated.size())
        cv::resize(mask, maskResized, annotated.size(), 0, 0, cv::INTER_NEAREST);
    else
        maskResized = mask.clone();

    cv::cvtColor(maskResized, maskColor, cv::COLOR_GRAY2BGR);

    cv::Mat combined;
    cv::hconcat(annotated, maskColor, combined);

    cv::imwrite(outPath, combined);
}


bool processEyeImage(const string& path,
                     double& biou,
                     const string& outDir)
{
    Mat eye = imread(path);
    if (eye.empty()) return false;

    Mat norm = normalizeEyeCrop(eye);
    Mat gray; cvtColor(norm, gray, COLOR_BGR2GRAY);

    Mat mask;
    Point center; int radius;

    if (!findPupilMask(gray, mask, center, radius))
        return false;

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;

    biou = computeBIoU(mask, contours[0]);

    string base = fs::path(path).stem().string();

    saveEyeAndMaskSeparately(norm, mask, outDir, base);

    return true;
}


bool processFaceImage(const string& path,
                      double& biou,
                      const string& outPath)
{
    Mat left, right;
    std::vector<Point> leftPts, rightPts;
    if (!extractEyesFromFace(path, left, right, leftPts, rightPts))
        return false;

    double L = -1, R = -1;
    Mat mL, mR;

    // LEFT
    Mat gL; cvtColor(left, gL, COLOR_BGR2GRAY);
    Point cL; int rL;
    if (findPupilMask(gL, mL, cL, rL)) {
        vector<vector<Point>> cnt;
        findContours(mL, cnt, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (!cnt.empty())
            L = computeBIoU(mL, cnt[0]);
    }

    // RIGHT
    Mat gR; cvtColor(right, gR, COLOR_BGR2GRAY);
    Point cR; int rR;
    if (findPupilMask(gR, mR, cR, rR)) {
        vector<vector<Point>> cnt;
        findContours(mR, cnt, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (!cnt.empty())
            R = computeBIoU(mR, cnt[0]);
    }

    if (L < 0 && R < 0) return false;

    Mat chosenEye, chosenMask;
    vector<Point> chosenLandmarks;

    if (L >= R) {
        biou = L;
        chosenEye = left;
        chosenMask = mL;
        chosenLandmarks = leftPts;
    }
    else {
        biou = R;
        chosenEye = right;
        chosenMask = mR;
        chosenLandmarks = rightPts;
    }

    saveFaceAnnotatedResult(chosenEye,
                            chosenMask,
                            chosenLandmarks,
                            outPath,
                            biou);

    return true;
}


bool processVideo(const string& path,
                  double& biou,
                  string outPath)
{
    VideoCapture cap(path);
    if (!cap.isOpened()) return false;

    Mat frame, lastEye, lastMask;
    double sum = 0;
    int valid = 0;

    for (int i = 0; i < 5; i++) {
        if (!cap.read(frame)) break;

        imwrite("__tmp.jpg", frame);

        Mat left, right;
        std::vector<Point> leftPts, rightPts;
        if (!extractEyesFromFace("__tmp.jpg", left, right, leftPts, rightPts))
            continue;

        Mat g; cvtColor(left, g, COLOR_BGR2GRAY);
        Mat m; Point c; int r;

        if (findPupilMask(g, m, c, r)) {
            vector<vector<Point>> cnt;
            findContours(m, cnt, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (!cnt.empty()) {
                double v = computeBIoU(m, cnt[0]);
                sum += v;
                valid++;

                lastEye = left.clone();
                lastMask = m.clone();
            }
        }
    }

    if (valid == 0) return false;

    biou = sum / valid;
    saveResultImage(lastEye, lastMask, outPath, biou);

    return true;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        cerr << "Usage: ./app imageDataset\n";
        return 1;
    }

    string root = argv[1];
    string outRoot = "results";

    ofstream csv("biou_results.csv");
    csv << "Filename,Type,BIoU\n";

    int total = 0, correct = 0;

    cout << left << setw(30) << "Filename"
         << setw(12) << "Type"
         << setw(10) << "BIoU"
         << setw(12) << "Correct\n";
    cout << string(64, '-') << endl;

    for (string type : {"real", "synthetic"}) {
        for (string mode : {"eye", "face", "video"}) {

            string inDir  = root + "/" + type + "/" + mode;
            string outDir = outRoot + "/" + type + "/" + mode;
            fs::create_directories(outDir);

            if (!fs::exists(inDir)) continue;

            for (auto& f : fs::directory_iterator(inDir)) {

                string path = f.path().string();
                string name = f.path().stem().string();
                string outPath = outDir + "/" + name + "_result.jpg";

                double biou = -1;
                bool ok = false;

                if (mode == "eye" && isImageFile(path)){
                    ok = processEyeImage(path, biou, outDir);
                }else if (mode == "face" && isImageFile(path)){
                    ok = processFaceImage(path, biou, outPath);
                }else if (mode == "video" && isVideoFile(path)){
                    ok = processVideo(path, biou, outPath);
                }
                if (!ok) continue;

                bool isCorrect =
                    (type == "real"      && biou > 0.5) ||
                    (type == "synthetic" && biou < 0.5);

                total++;
                if (isCorrect) correct++;

                csv << f.path().filename().string() << ","
                    << type << ","
                    << biou << "\n";

                cout << setw(30) << f.path().filename().string()
                     << setw(17) << type+"|"+mode
                     << setw(10) << fixed << setprecision(3) << biou
                     << setw(12) << (isCorrect ? "YES" : "NO") << endl;
            }
        }
    }

    csv.close();

    double accuracy = total ? (double)correct / total : 0;

    cout << "\n========================================\n";
    cout << "TOTAL FILES  : " << total << endl;
    cout << "CORRECT      : " << correct << endl;
    cout << "FINAL ACCURACY = " << accuracy << endl;
    cout << "========================================\n";

    return 0;
}
