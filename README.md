# imageforensics

# Pupil Segmentation Pipeline

**Preprocess | Localization | CAHT Center Detection | Post-Processing**

This repository implements a pupil-segmentation pipeline built around
CAHT (Circular/Axial Hough Transform).
The system is divided into four major stages:

1.  **Preprocess**
2.  **Pupil Localization**
3.  **CAHT-Based Center Identification**
4.  **Post-Processing (Mask Refinement + BIoU Computation)**


## 1. Preprocess

### `preprocessEyeImage()`

**Purpose:**
Prepare the input eye image for accurate edge extraction and shape
analysis.

**Typical operations:**
- Grayscale conversion
- Intensity normalization
- MEdian Blur filtering
- Cropping/resizing depending on source


## 2. Pupil Localization
### `extractEyesFromFace( const string& imagePath, cv::Mat& leftEye, cv::Mat& rightEye, std::vector<cv::Point>& leftLandmarks, std::vector<cv::Point>& rightLandmarks)`
Extracts the eye from a given face, if the input file is of type face. If the input file is an eye, it performs localization operations and identifies an eyebox by either cropping or padding to the box.
Choosing the darkest spot in the given eye segmented from the entire face.

## 3. CAHT-Based Center Identification
Primary pipeline function.
### `findPupilMask()`
**Outputs:**
- `pupilMask`
- `center`
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

#### 3. CAHT Voting Loop (Center Detection)This step performs the standard Hough voting, where every edge pixel contributes votes to potential circle centers.
This document describes the logic used for computing a pupil mask based
on circle candidates in an image.

##### Compute Mean Intensity Inside the Circle

This checks how dark the region is.

##### Crop the bounding box

``` cpp
int x0 = std::max(0, cpt.x - r);
int y0 = std::max(0, cpt.y - r);
int x1 = std::min(I.cols-1, cpt.x + r);
int y1 = std::min(I.rows-1, cpt.y + r);
Rect roi(x0,y0,x1-x0+1,y1-y0+1);
if (roi.width <=0 || roi.height <=0) continue;
Mat patch = I(roi);
```

`patch` = sub-image containing the entire candidate circle.

##### Create a circular mask inside that patch

``` cpp
Mat circMask = Mat::zeros(patch.size(), CV_8UC1);
circle(circMask, Point(cpt.x - x0, cpt.y - y0), r, Scalar(255), FILLED);
```

This mask selects only pixels inside the circle.

##### Compute mean intensity

``` cpp
Scalar meanInside = mean(patch, circMask);
double meanVal = meanInside[0];
```

-   Dark region implies lower meanVal\
-   Bright region implies higher meanVal


##### Edge coverage along circle boundary

This checks how well edges align with the circle boundary.

##### Sample many points along the circumference

``` cpp
int N = std::max(20, r);
int edgeCount = 0;
for (int k = 0; k < N; k++) {
    double a = 2.0 * CV_PI * k / N;
    int sx = cvRound(cpt.x + r * cos(a));
    int sy = cvRound(cpt.y + r * sin(a));
```

##### Count how many points fall on edges

(For example, from a Canny detector)

``` cpp
if (edges.at<uchar>(sy,sx) > 0) edgeCount++;
```

##### Compute edge coverage fraction

``` cpp
double edgeCoverage = double(edgeCount) / double(N);
```

-   High edgeCoverage - circle boundary matches real edges\
-   Low edgeCoverage - likely noise or incorrect circle


##### Compute Combined Score

``` cpp
double score = (255.0 - meanVal) * 0.6 + edgeCoverage * 255.0 * 0.4;
```

##### Breakdown:

-   `(255 - meanVal)` - darker circles score higher\
-   `edgeCoverage * 255` - strong edge alignment scores higher\
-   Weights:
    -   0.6 for darkness\
    -   0.4 for edge alignment


##### Penalty for Distance from Image Center

Encourages selecting circles near the eye's center.

``` cpp
double cx = I.cols/2.0, cy = I.rows/2.0;
double dist = sqrt((cpt.x-cx)*(cpt.x-cx) + (cpt.y-cy)*(cpt.y-cy));
double distPenalty = std::max(0.0, dist - std::min(I.cols,I.rows)/4.0);
score -= distPenalty * 0.05;
```

-   Inside central quarter - no penalty\
-   Farther away - penalized

##### Select Best Circle

``` cpp
if (score > bestScore) {
    bestScore = score;
    bestC = c;
}
```
The chosen circle is the one with the highest final score.

Morphological operations are applied afterwards. 
The expression 
### `Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3))`
constructs a small elliptical kernel used for dilation or erosion. 
### `morphologyEx(edges, edges, MORPH_CLOSE, kernel)`
performs a closing operation dilation followed by erosion which closes small gaps along edges. 
A similar call with MORPH_OPEN removes thin streaks. These steps ensure CAHT’s custom streak removal operation.

### Build Pupil MaskFinally, a binary mask of the detected pupil is generated using the determined center and radius.
mask = cv::Mat::zeros(I.size(), CV_8UC1);
cv::circle(mask, center, radius, 255, FILLED);


## 4. Post-Processing

### `buildPupilMaskFromCenter()`

Region growing + morphological smoothing.

Computes Boundary IoU for evaluation.


