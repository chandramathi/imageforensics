#include "PupilSegment.h"

using namespace cv;
using std::vector;

// Helper: normalize and denoise image similar to CAHT pre-step
static void preprocessForPupil(const Mat &in, Mat &out)
{
    // input: single channel
    Mat tmp;
    // Normalize bit
    in.convertTo(tmp, CV_8UC1); // 8U: Specifies an 8-bit unsigned integer data type (ranging from 0 to 255).
                                //  C1: Specifies that the matrix will have 1 channel, meaning it will be a single-channel grayscale image.
    // Contrast normalize (CLAHE helps with uneven lighting)
    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    clahe->apply(tmp, tmp);
    // median blur to reduce small specular highlights
    medianBlur(tmp, tmp, 5);
    out = tmp;
}

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
bool findPupilMask(const Mat &eyeGray, Mat &pupilMask, Point &center, int &radius, int cannyLow,
                   int cannyHigh, int houghMinR, int houghMaxR, double dp, int minDist, int houghParam1,
                   int houghParam2)
{
    if (eyeGray.empty() || eyeGray.channels() != 1)
        return false;

    Mat I;
    preprocessForPupil(eyeGray, I);

    // 1) Generate edge map (similar role as caht's canny -> thin -> accumulation).
    Mat edges;
    // Use Canny; CAHT uses a custom canny implementation, but Canny suffices here.
    Canny(I, edges, cannyLow, cannyHigh, 3);

    // 2) Some morphological cleanups (remove thin streaks similar to remove_streaks)
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);
    morphologyEx(edges, edges, MORPH_OPEN, kernel);

    // 3) Hough circle (OpenCV) to propose pupil candidates (CAHT uses its hough_circle)
    vector<Vec3f> circles;
    HoughCircles(I, circles, HOUGH_GRADIENT, dp, minDist, houghParam1, houghParam2, houghMinR, houghMaxR);

    if (circles.empty())
    {
        // fallback: try a more permissive parameter set
        HoughCircles(I, circles, HOUGH_GRADIENT, 1.0, minDist / 2, houghParam1 / 2, houghParam2 / 2, houghMinR / 2, houghMaxR * 2);
    }

    if (circles.empty())
        return false;

    // 4) choose best candidate: prefer darker region + strong edge coverage (analogous to find_best_circle)
    double bestScore = -1.0;
    Vec3f bestC;
    for (const auto &c : circles)
    {
        Point cpt(cvRound(c[0]), cvRound(c[1]));
        int r = cvRound(c[2]);
        // ignore invalid
        if (r <= 2)
            continue;
        // compute mean intensity inside circle (pupil should be dark)
        int x0 = std::max(0, cpt.x - r);
        int y0 = std::max(0, cpt.y - r);
        int x1 = std::min(I.cols - 1, cpt.x + r);
        int y1 = std::min(I.rows - 1, cpt.y + r);
        Rect roi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
        if (roi.width <= 0 || roi.height <= 0)
            continue;
        Mat patch = I(roi);
        // mask for circle
        Mat circMask = Mat::zeros(patch.size(), CV_8UC1);
        circle(circMask, Point(cpt.x - x0, cpt.y - y0), r, Scalar(255), FILLED);
        Scalar meanInside = mean(patch, circMask);
        double meanVal = meanInside[0];

        // edge coverage: how many edge pixels around circle circumference (approx)
        // sample N points on circumference and check edges
        int N = std::max(20, r);
        int edgeCount = 0;
        for (int k = 0; k < N; k++)
        {
            double a = 2.0 * CV_PI * k / N;
            int sx = cvRound(cpt.x + r * cos(a));
            int sy = cvRound(cpt.y + r * sin(a));
            if (sx >= 0 && sx < edges.cols && sy >= 0 && sy < edges.rows)
            {
                if (edges.at<uchar>(sy, sx) > 0)
                    edgeCount++;
            }
        }
        double edgeCoverage = double(edgeCount) / double(N);

        // Score: darker inside + more edge coverage preferred
        double score = (255.0 - meanVal) * 0.6 + edgeCoverage * 255.0 * 0.4;
        // penalize circles outside central image area (CAHT uses hint weighting)
        double cx = I.cols / 2.0, cy = I.rows / 2.0;
        double dist = sqrt((cpt.x - cx) * (cpt.x - cx) + (cpt.y - cy) * (cpt.y - cy));
        double distPenalty = std::max(0.0, dist - std::min(I.cols, I.rows) / 4.0);
        score -= distPenalty * 0.05;

        if (score > bestScore)
        {
            bestScore = score;
            bestC = c;
        }
    }

    if (bestScore < 0)
        return false;

    center = Point(cvRound(bestC[0]), cvRound(bestC[1]));
    radius = cvRound(bestC[2]);

    // 5) produce mask (filled circle). Optionally refine mask using local thresholding
    pupilMask = Mat::zeros(I.size(), CV_8UC1);
    circle(pupilMask, center, radius, Scalar(255), FILLED);

    //Specular highlight removal
    {
        // Extract local ROI around pupil
        int x0 = std::max(0, center.x - radius);
        int y0 = std::max(0, center.y - radius);
        int w = std::min(radius * 2 + 1, I.cols - x0);
        int h = std::min(radius * 2 + 1, I.rows - y0);
        Rect roi(x0, y0, w, h);

        Mat local = I(roi);

        // Mask of the pupil region inside ROI
        Mat localPupilMask = Mat::zeros(local.size(), CV_8UC1);
        circle(localPupilMask, Point(radius, radius), radius, Scalar(255), FILLED);

        // Extract pixel values inside the pupil
        Mat pupilVals;
        local.copyTo(pupilVals, localPupilMask);

        // Threshold to find bright pixels (specular highlights)
        double t = threshold(pupilVals, pupilVals, 0, 255, THRESH_BINARY | THRESH_OTSU);

        // Highlights are pixels > t
        Mat highlights = (local > t + 10) & localPupilMask;

        // Morphological operations to clean noise
        morphologyEx(highlights, highlights, MORPH_OPEN,
                     getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

        // Connected component analysis
        Mat labels, stats, centroids;
        int nCC = connectedComponentsWithStats(highlights, labels, stats, centroids);

        for (int i = 1; i < nCC; i++)
        { // skip label 0 (background)
            int area = stats.at<int>(i, CC_STAT_AREA);
            int left = stats.at<int>(i, CC_STAT_LEFT);
            int top = stats.at<int>(i, CC_STAT_TOP);
            int w = stats.at<int>(i, CC_STAT_WIDTH);
            int h = stats.at<int>(i, CC_STAT_HEIGHT);

            // Heuristic: specular highlights are small
            // If needed, change max area depending on image resolution.
            if (area < 300)
            {
                // Remove highlight → fill region in the pupil mask
                for (int yy = top; yy < top + h; yy++)
                {
                    for (int xx = left; xx < left + w; xx++)
                    {
                        if (labels.at<int>(yy, xx) == i)
                        {
                            pupilMask.at<uchar>(y0 + yy, x0 + xx) = 255; // restore
                        }
                    }
                }
            }
        }
    }
    // Create small mask of inside, apply Otsu on that area to separate specular spots
    Rect roi(max(0, center.x - radius), max(0, center.y - radius),
             min(radius * 2 + 1, I.cols - (center.x - radius)),
             min(radius * 2 + 1, I.rows - (center.y - radius)));
    if (roi.width > 10 && roi.height > 10)
    {
        Mat local = I(roi);
        Mat localMask = Mat::zeros(local.size(), CV_8UC1);
        circle(localMask, Point(local.cols / 2, local.rows / 2), radius, Scalar(255), FILLED);
        Mat localVals;
        local.copyTo(localVals, localMask);
        // Otsu inside pupil region: if there are bright speculars, this will separate.
        double t = threshold(localVals, localVals, 0, 255, THRESH_BINARY | THRESH_OTSU);
        // Remove bright spots inside pupil if any: pixels above Otsu inside circle -> not pupil
        for (int y = 0; y < local.rows; y++)
        {
            for (int x = 0; x < local.cols; x++)
            {
                if (localMask.at<uchar>(y, x) && local.at<uchar>(y, x) > t + 10)
                {
                    // erase small region
                    circle(pupilMask, Point(roi.x + x, roi.y + y), 2, Scalar(0), FILLED);
                }
            }
        }
    }

    // final morphological clean
    morphologyEx(pupilMask, pupilMask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    morphologyEx(pupilMask, pupilMask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

    // sanity check: ensure mask area is reasonable
    double area = countNonZero(pupilMask);
    if (area < 10.0)
        return false;
    return true;
}
