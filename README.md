# Pattern Recognition Study Assignment

## 1. Panorama Stitching

### Overview
Implementation of a complete panorama stitching pipeline using computer vision techniques.

### Key Components
- **Harris Corner Detection**
  - Sobel operators for gradient computation
  - Second moment matrix calculation
  - Non-maximum suppression implementation
  - Example result: [](@replace=1)


- **Feature Matching**
  - HOG and SIFT descriptors
  - L2 distance matching
  - Mutual nearest neighbor filtering
  - Matching results: [](@replace=2)


- **Image Stitching**
  - RANSAC for affine transformation
  - Image warping and blending
  - Boundary handling

## 2. Graph-Based Segmentation & SLIC

### Graph-Based Segmentation
- Graph construction from pixels
- Efficient region merging
- Color and texture feature handling

### SLIC Superpixels
- Simple Linear Iterative Clustering
- Regular, compact superpixel generation
- Performance advantages

## 3. Semi-Supervised Image Classification

### FixMatch Implementation
- **Model Architecture**
  - WideResNet-28-2 backbone
  - Dual data augmentation (weak + strong)
  - Confidence-based pseudo-labeling

- **Experimental Results**
  | Labeled Samples | Best Accuracy | Optimal Epoch |
  |----------------|---------------|---------------|
  | 40             | 18.44%        | 4             |
  | 250            | 22.73%        | 5             |
  | 4000           | 55.27%        | 2             |

- **Training Curves**
  - 40 samples: [](@replace=3)

  - 250 samples: [](@replace=4)

  - 4000 samples: [](@replace=5)


### Optimizations
- Checkpoint saving mechanism
- Early stopping implementation
- Training visualization
- High-confidence pseudo-label filtering

## Repository
GitHub: [bobbyjin00c/pattern_recgonition_SA_Spring25](https://github.com/bobbyjin00c/pattern_recgonition_SA_Spring25)
