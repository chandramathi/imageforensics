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
Morphological operations are applied afterwards. 
The expression 
### `Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3))`
constructs a small elliptical kernel used for dilation or erosion. 
### `morphologyEx(edges, edges, MORPH_CLOSE, kernel)`
performs a closing operation dilation followed by erosion which closes small gaps along edges. 
A similar call with MORPH_OPEN removes thin streaks. These steps ensure CAHT’s custom streak removal operation.



## 4. Post-Processing

### `buildPupilMaskFromCenter()`

Region growing + morphological smoothing.

Computes Boundary IoU for evaluation.


