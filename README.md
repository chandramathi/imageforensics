# imageforensics

# Pupil Segmentation Pipeline

**Preprocess | Localization | CAHT Center Detection | Post-Processing**

This repository implements a pupil-segmentation pipeline built around
CAHT (Circular/Axial Hough Transform).\
The system is divided into four major stages:

1.  **Preprocess**\
2.  **Pupil Localization**\
3.  **CAHT-Based Center Identification**\
4.  **Post-Processing (Mask Refinement + BIoU Computation)**


## 1. Preprocess

### `preprocessEyeImage()`

**Purpose:**\
Prepare the input eye image for accurate edge extraction and shape
analysis.

**Typical operations:**\
- Grayscale conversion\
- Intensity normalization\
- MEdian Blur filtering\
- Cropping/resizing depending on source


## 2. Pupil Localization
### `extractEyesFromFace( const string& imagePath, cv::Mat& leftEye, cv::Mat& rightEye, std::vector<cv::Point>& leftLandmarks, std::vector<cv::Point>& rightLandmarks)`
Extracts the eye from a given face, if the input file is of type face. If the input file is an eye, it performs localization operations and identifies an eyebox by either cropping or padding to the box.
Choosing the darkest spot in the given eye segmented from the entire face.

## 3. CAHT-Based Center Identification
Primary pipeline function.
### `findPupilMask()`
**Outputs:**\
- `pupilMask`\
- `center`\
- `radius`

### `Canny(I, edges, low, high, 3)` 
computes intensity gradients and marks strong and weak edge pixels. The parameters low and high define thresholds for edge detection. The number 3 indicates the aperture size used for gradient calculation.
### Circular Accumulator Hough Transform (CAHT) for Pupil Detection

This document outlines the steps for implementing a **Circular Accumulator Hough Transform (CAHT)** to detect a pupil in an image, focusing on the simplification from a 3D to a 2D accumulator space.

#### 1. Data Structures

The core input is an image, typically pre-processed to find edges.

| Variable | Type | Description |
| :--- | :--- | :--- |
| `rows`, `cols` | `int` | Dimensions of the input image. |
| `acc` | `cv::Mat` (`CV_32SC1`) | The 2D Accumulator array. Stores vote counts for potential circle centers. Initialized to zero. |
| `rMin`, `rMax` | `int` | The minimum and maximum radius to search for. |
| `edges` | `cv::Mat` (`CV_8UC1`) | The binary edge map from Canny ). |
| `eRow` | `uchar*` | Pointer to a row of the `edges` matrix (for fast access). |


#### 2. Initialize Accumulator

`cv::Mat acc = cv::Mat::zeros(rows, cols, CV_32SC1);`

#### 3. CAHT Voting Loop (Center Detection)This step performs the standard Hough voting, where every edge pixel contributes votes to potential circle centers.C++
// For each radius:
for (int r = rMin; r <= rMax; r++)
{
    // Iterate through all edge pixels (x, y)
    for (int y = 0; y < rows; y++)
    {
        uchar* eRow = edges.ptr<uchar>(y); // Get the edge row data
        for (int x = 0; x < cols; x++)
        {
            // For each edge pixel:
            if (!eRow[x]) continue;

            // For each angle on that radius (usually 0 to 360 degrees, or a simplified subset):
            // (Note: The angle loop is often implicitly handled by pre-calculating a table or iterating over sampled points)
            
            // Assuming 'ang' is the angle of the gradient or a sampled angle around the point (x, y)
            // The coordinates of the potential center (cx, cy) are calculated:
            int cx = cvRound(x - r * cos(ang));
            int cy = cvRound(y - r * sin(ang));

            // ...

            // Vote into the accumulator:
            acc.at<int>(cy, cx)++;
        }
    }
}

The Voting Equation - The equation cx = cvRound(x - r * cos(ang)) and cy = cvRound(y - r * sin(ang)) is based on the circle formula:$$(x - c_x)^2 + (y - c_y)^2 = r^2$$If an edge point $(x, y)$ lies on a circle of radius $r$, the center $(c_x, c_y)$ must be located at a distance $r$ from $(x, y)$ in the direction opposite to the angle $\theta$ ($\text{ang}$), which often relates to the local edge gradient direction. Find the Best Center After the voting loop, the 2D accumulator contains the total support for every possible center location. The location with the maximum votes is the most likely circle center. Iterate through the accumulator 'acc' to find the maximum value (maxVal) and its coordinates (bestCenter)
if (row[x] > maxVal)
{
    maxVal = row[x];
    bestCenter = {x, y};
}
The location with the most votes is the center of the circle that had the most edge support across all radii $\rightarrow$ likely the pupil center.5. Estimate Radius (Post-Center Fixation)With the bestCenter fixed, the algorithm now searches for the best radius $r$ by sampling along the circumference of a circle centered at bestCenter.C++// center is now fixed as 'bestCenter'
for (int r = rMin; r <= rMax; r++)
{
    // For each radius:
    // Sample the circumference of the circle (center, r)
    // Count how many samples land on edge pixels
    // ...
    if (edges.at<uchar>(py,px)) // py, px are the sampled coordinates on the circumference
        votes++;

    // Choose the radius with the most edge support (not explicitly shown, but implied by the logic)
}
// This gives a robust radius estimate and sets the final 'radius'.

Morphological operations are applied afterwards. 
The expression 
### `Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3))`
constructs a small elliptical kernel used for dilation or erosion. 
### `morphologyEx(edges, edges, MORPH_CLOSE, kernel)`
performs a closing operation dilation followed by erosion which closes small gaps along edges. 
A similar call with MORPH_OPEN removes thin streaks. These steps ensure CAHT’s custom streak removal operation.

### Build Pupil MaskFinally, a binary mask of the detected pupil is generated using the determined center and radius.C++mask = cv::Mat::zeros(I.size(), CV_8UC1);
cv::circle(mask, center, radius, 255, FILLED);


## 4. Post-Processing

### `buildPupilMaskFromCenter()`

Region growing + morphological smoothing.

Computes Boundary IoU for evaluation.


