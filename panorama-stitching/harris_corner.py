import cv2
import numpy as np

def harris_corner_detection(input_path, output_path,
                            block_size=4, ksize=5, k=0.04, thresh=0.07):
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, np.ones((3,3), np.uint8))
    keypoints = np.argwhere(dst > thresh * dst.max())
    for y, x in keypoints:
        img[y, x] = (0, 255, 0)  # BGR
    cv2.imwrite(output_path, img)

if __name__ == '__main__':
    harris_corner_detection('images/sudoku.png', 'results/1/sudoku_keypoints.png')
