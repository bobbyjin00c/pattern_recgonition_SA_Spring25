# stitching_1.py
import os
import glob
import cv2
import numpy as np

def stitch_pair(img1, img2, kp1, kp2, matches, method='homography'):
 
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)


    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)


    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)
    all_corners = np.concatenate([
        warped_corners,
        np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    ], axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translate = [-xmin, -ymin]
    T = np.array([
        [1, 0, translate[0]],
        [0, 1, translate[1]],
        [0, 0, 1]
    ])

    pano_width  = xmax - xmin
    pano_height = ymax - ymin

    pano = cv2.warpPerspective(img2, T @ H, (pano_width, pano_height))
    pano[translate[1]:translate[1]+h1, translate[0]:translate[0]+w1] = img1

    return pano

def batch_stitch(image_paths, output_path):
    imgs = [cv2.imread(p) for p in image_paths]
    imgs = [img for img in imgs if img is not None]

    for p in image_paths:
        print("    ", p)

    sift = cv2.SIFT_create()
    pano = imgs[0]

    for idx, img in enumerate(imgs[1:], start=2):
        gray1 = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = sift.detectAndCompute(gray1, None)
        kp2, desc2 = sift.detectAndCompute(gray2, None)
        bf = cv2.BFMatcher()
        knn = bf.knnMatch(desc1, desc2, k=2)
        good = [m for m,n in knn if m.distance < 0.75 * n.distance]

        pano = stitch_pair(pano, img, kp1, kp2, good, method='homography')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, pano)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, 'images', '1')
    image_list = sorted(glob.glob(os.path.join(image_dir, 'yosemite*.jpg')))
    output_path = os.path.join(base_dir, 'results', '1', 'yosemite_stitching.png')
    batch_stitch(image_list, output_path)
