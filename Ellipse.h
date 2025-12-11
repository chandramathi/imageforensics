
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>   

using namespace cv;

/**
 * CustomEllipseFitter
 *
 * This is a custom class for performing ellipse fitting using eigen value decomposition
 *  of a Direct Least Squares (DLS) ellipse fitter. 
 * All linear-algebra matrices uses OpenCV `Mat` calls
 * (Mat double-precision).
 * For 2x2 and 3x3 ops we rely on invert, eigen and basic Mat math.
 */
class CustomEllipseFitter {
private:
    // Helper function - create zero matrix of size r x c (CV_64F)
    static Mat zeros(int r, int c) {
        return Mat::zeros(r, c, CV_64F);
    }

    // Helper function -  safe inverse for small matrix (uses invert). throws/returns false on failure.
    static bool invertMat(const Mat &src, Mat &dst) {
        if (src.empty()) return false;
        double det = determinant(src);
        if (std::abs(det) < 1e-12) {
            return false;
        }
        return invert(src, dst, DECOMP_LU) != 0;
    }

    // Helper function - extract block (r0..r1-1, c0..c1-1)
    static Mat block(const Mat &M, int r0, int r1, int c0, int c1) {
        return M(Range(r0, r1), Range(c0, c1)).clone();
    }

    // Helper function - convert vector<double> row into Mat(col)
    static Mat colVec(const std::vector<double> &v) {
        Mat m((int)v.size(), 1, CV_64F);
        for (size_t i = 0; i < v.size(); ++i) m.at<double>((int)i, 0) = v[i];
        return m;
    }

    // conic (A,B,C,D,E,F) - RotatedRect (de-normalized by center_shift and scale)
    static RotatedRect conicToEllipse(const Mat &coef /*6x1*/, const Point2f &center_shift, double scale) {
        // coef must be 6x1 CV_64F
        double A = coef.at<double>(0, 0);
        double B = coef.at<double>(1, 0);
        double C = coef.at<double>(2, 0);
        double D = coef.at<double>(3, 0);
        double E = coef.at<double>(4, 0);
        double F = coef.at<double>(5, 0);

        // 1) center: solve  [2A  B ] [cx] = [-D]
        //                  [ B 2C] [cy]   [-E]
        Mat Qc = zeros(2, 2);
        Qc.at<double>(0, 0) = 2.0 * A;
        Qc.at<double>(0, 1) = B;
        Qc.at<double>(1, 0) = B;
        Qc.at<double>(1, 1) = 2.0 * C;

        Mat Vc = zeros(2, 1);
        Vc.at<double>(0, 0) = -D;
        Vc.at<double>(1, 0) = -E;

        Mat center_norm = zeros(2, 1);
        if (!invertMat(Qc, Qc)) {
            return RotatedRect();
        }
        center_norm = Qc * Vc;
        double cx_norm = center_norm.at<double>(0, 0);
        double cy_norm = center_norm.at<double>(1, 0);

        // 2) shifted constant term
        double F_shifted = A*cx_norm*cx_norm + B*cx_norm*cy_norm + C*cy_norm*cy_norm
                           + D*cx_norm + E*cy_norm + F;

        // 3) quadratic form matrix for axes/eigen
        Mat Qm = zeros(2, 2);
        Qm.at<double>(0, 0) = A;
        Qm.at<double>(0, 1) = B/2.0;
        Qm.at<double>(1, 0) = B/2.0;
        Qm.at<double>(1, 1) = C;

        // eigen decomposition of symmetric 2x2 Qm
        Mat evals2, evecs2;
        if (!eigen(Qm, evals2, evecs2)) {
            return RotatedRect();
        }
        // eigen returns eigenvalues in descending order and eigenvectors as rows.
        // eigenvalue i corresponds to row i of evecs2.
        // But we want lambda(0), lambda(1) consistent with previous code.
        double lambda0 = evals2.at<double>(0, 0);
        double lambda1 = evals2.at<double>(1, 0);

        // calculate semiaxes squared (den = -F_shifted)
        double den = -F_shifted;
        if (std::abs(lambda0) < 1e-18 || std::abs(lambda1) < 1e-18 || den <= 0) {
            // invalid ellipse (degenerate)
            return RotatedRect();
        }

        double a_sq = den / lambda0;
        double b_sq = den / lambda1;
        if (a_sq < 0) a_sq = std::abs(a_sq); 
        if (b_sq < 0) b_sq = std::abs(b_sq);

        double a_half = std::sqrt(a_sq);
        double b_half = std::sqrt(b_sq);

        // pick angle: eigenvectors are rows; choose corresponding eigenvector
        Mat v0 = evecs2.row(0).t(); // 2x1 col
        Mat v1 = evecs2.row(1).t();
        double angle_rad;
        if (std::abs(lambda0) < std::abs(lambda1)) {
            angle_rad = std::atan2(v0.at<double>(1,0), v0.at<double>(0,0));
        } else {
            angle_rad = std::atan2(v1.at<double>(1,0), v1.at<double>(0,0));
            std::swap(a_half, b_half);
        }

        RotatedRect box;
        // de-normalize center
        box.center.x = static_cast<float>(cx_norm / scale + center_shift.x);
        box.center.y = static_cast<float>(cy_norm / scale + center_shift.y);

        // axes: OpenCV store size.width = major axis length
        box.size.width  = static_cast<float>(2.0 * b_half / scale);
        box.size.height = static_cast<float>(2.0 * a_half / scale);

        // angle in degrees
        box.angle = static_cast<float>(angle_rad * 180.0 / M_PI);
        while (box.angle < 0) box.angle += 180.0f;
        while (box.angle >= 180.0) box.angle -= 180.0f;

        // ensure width corresponds to angle convention (match original logic)
        if (box.size.width > box.size.height) {
            std::swap(box.size.width, box.size.height);
            box.angle += 90.0f;
            if (box.angle >= 180.0f) box.angle -= 180.0f;
        }

        return box;
    }

public:
    // Fit returns an OpenCV RotatedRect. Contour must have at least 5 points.
    RotatedRect fit(const std::vector<Point> &contour) {
        if (contour.size() < 5) {
            std::cerr << "Error: At least 5 points are required for ellipse fitting." << std::endl;
            return RotatedRect();
        }

        const int N = (int)contour.size();

        // 1) centroid & scale for normalization
        Point2f centroid(0.0f, 0.0f);
        for (const auto &p : contour) {
            centroid.x += p.x;
            centroid.y += p.y;
        }
        centroid.x /= N;
        centroid.y /= N;

        double s = 0.0;
        for (const auto &p : contour) {
            s += std::abs(p.x - centroid.x) + std::abs(p.y - centroid.y);
        }
        double scale = 100.0 / (s > 1e-8 ? s : 1e-8);

        // 2) build design matrix D (Nx6) and scatter matrix S = D^T * D (6x6)
        Mat D = zeros(N, 6);
        for (int i = 0; i < N; ++i) {
            double x = (contour[i].x - centroid.x) * scale;
            double y = (contour[i].y - centroid.y) * scale;
            D.at<double>(i, 0) = x * x;
            D.at<double>(i, 1) = x * y;
            D.at<double>(i, 2) = y * y;
            D.at<double>(i, 3) = x;
            D.at<double>(i, 4) = y;
            D.at<double>(i, 5) = 1.0;
        }
        Mat S = D.t() * D; // 6x6

        // 3) Constraint matrix C (6x6)
        Mat C = zeros(6, 6);
        C.at<double>(0, 2) = 2.0;
        C.at<double>(1, 1) = -1.0;
        C.at<double>(2, 0) = 2.0;

        // 4) Partition S into blocks
        Mat S11 = block(S, 0, 3, 0, 3); // 3x3
        Mat S12 = block(S, 0, 3, 3, 6); // 3x3
        Mat S21 = block(S, 3, 6, 0, 3); // 3x3
        Mat S22 = block(S, 3, 6, 3, 6); // 3x3

        // invert S22
        Mat S22_inv;
        if (!invertMat(S22, S22_inv)) {
            std::cerr << "S22 is singular; ellipse fit failed." << std::endl;
            return RotatedRect();
        }

        // T = S11 - S12 * S22^{-1} * S21
        Mat T = S11 - S12 * S22_inv * S21;

        // C3 (3x3) the reduced constraint
        Mat C3 = zeros(3, 3);
        C3.at<double>(0, 2) = 2.0;
        C3.at<double>(2, 0) = 2.0;
        C3.at<double>(1, 1) = -1.0;

        // invert C3
        Mat C3_inv;
        if (!invertMat(C3, C3_inv)) {
            std::cerr << "C3 singular; cannot continue." << std::endl;
            return RotatedRect();
        }

        // M = C3_inv * T
        Mat M = C3_inv * T;

        // 5) solve eigenproblem for M (3x3). eigen returns eigenvalues (nx1) and
        // eigenvectors as rows (n x n), matching each eigenvalue to corresponding row.
        Mat evals3, evecs3;
        if (!eigen(M, evals3, evecs3)) {
            std::cerr << "Eigen decomposition (3x3) failed." << std::endl;
            return RotatedRect();
        }

        // choose eigenvector q satisfying ellipse condition:
        // det(4*q0*q2 - q1^2) > 0 and eigenvalue > 0, take minimal eigenvalue among those
        Mat q_vec = zeros(3,1);
        double min_val = std::numeric_limits<double>::infinity();
        bool found = false;
        // Note: eigen returns eigenvalues in descending order (largest first)
        for (int i = 0; i < 3; ++i) {
            double eigval = evals3.at<double>(i, 0);
            // eigenvector is row i of evecs3 (1x3). we need it as column vector
            Mat q_candidate = evecs3.row(i).t(); // 3x1

            double q0 = q_candidate.at<double>(0,0);
            double q1 = q_candidate.at<double>(1,0);
            double q2 = q_candidate.at<double>(2,0);

            double det_cond = 4.0 * q0 * q2 - q1 * q1;
            if (det_cond > 0 && eigval > 0) {
                if (eigval < min_val) {
                    min_val = eigval;
                    q_vec = q_candidate.clone();
                    found = true;
                }
            }
        }

        if (!found) {
            // if no valid vector found, fallback: try to pick the eigenvector with positive det (if any)
            for (int i = 0; i < 3; ++i) {
                Mat q_candidate = evecs3.row(i).t();
                double q0 = q_candidate.at<double>(0,0);
                double q1 = q_candidate.at<double>(1,0);
                double q2 = q_candidate.at<double>(2,0);
                double det_cond = 4.0 * q0 * q2 - q1 * q1;
                if (det_cond > 0) {
                    q_vec = q_candidate.clone();
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "No valid conic (ellipse) eigenvector found." << std::endl;
                return RotatedRect();
            }
        }

        // 6) compute linear part r = S22^{-1} * S21 * q
        Mat r = S22_inv * (S21 * q_vec); // 3x1

        // 7) combine coefficients: coef = [q; -r]
        Mat coef = zeros(6, 1);
        coef.at<double>(0,0) = q_vec.at<double>(0,0);
        coef.at<double>(1,0) = q_vec.at<double>(1,0);
        coef.at<double>(2,0) = q_vec.at<double>(2,0);
        coef.at<double>(3,0) = -r.at<double>(0,0);
        coef.at<double>(4,0) = -r.at<double>(1,0);
        coef.at<double>(5,0) = -r.at<double>(2,0);

        // 8) convert to RotatedRect and denormalize
        return conicToEllipse(coef, centroid, scale);
    }
};
