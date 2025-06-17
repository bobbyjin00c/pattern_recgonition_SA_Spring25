# descriptor_matching.py
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure

def extract_harris_keypoints(img, max_corners=200, quality_level=0.03, min_distance=12):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
    pts = np.int0(corners).reshape(-1, 2)
    return [cv2.KeyPoint(float(x), float(y), 3) for x, y in pts]

def compute_hog_descriptors(img, keypoints, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descs = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = gray[y-16:y+16, x-16:x+16]
        if patch.shape[0] < 32 or patch.shape[1] < 32:
            descs.append(np.zeros((cells_per_block[0]*cells_per_block[1]*9,)))
            continue
        h, _ = hog(patch,
                   orientations=9,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm='L2-Hys',
                   visualize=True)
        descs.append(h)
    return np.array(descs, dtype=np.float32)

def match_and_draw(img1, kp1, desc1, img2, kp2, desc2, output_path,
                   descriptor_type='SIFT', top_k=50):
    if descriptor_type == 'SIFT':
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)[:top_k]

    out = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(output_path, out)

if __name__ == '__main__':

    img1 = cv2.imread('images/1/uttower1.jpg')
    img2 = cv2.imread('images/1/uttower2.jpg')

    ksize = 3
    kp1 = extract_harris_keypoints(img1, max_corners=200, quality_level=0.03, min_distance=12)
    kp2 = extract_harris_keypoints(img2, max_corners=200, quality_level=0.03, min_distance=12)

    img1_kp = img1.copy()
    img2_kp = img2.copy()
    for kp in kp1:
        cv2.circle(img1_kp, (int(kp.pt[0]), int(kp.pt[1])), radius=2, color=(0,255,0), thickness=-1)
    for kp in kp2:
        cv2.circle(img2_kp, (int(kp.pt[0]), int(kp.pt[1])), radius=2, color=(0,255,0), thickness=-1)

    cv2.imwrite('results/1/uttower1_keypoints.jpg', img1_kp)
    cv2.imwrite('results/1/uttower2_keypoints.jpg', img2_kp)

    # SIFT 描述与匹配
    sift = cv2.SIFT_create()
    desc1_sift = sift.compute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), kp1)[1]
    desc2_sift = sift.compute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), kp2)[1]
    match_and_draw(img1, kp1, desc1_sift, img2, kp2, desc2_sift,
                   'results/1/uttower_match_sift.png', descriptor_type='SIFT')

    # HOG 描述与匹配
    desc1_hog = compute_hog_descriptors(img1, kp1)
    desc2_hog = compute_hog_descriptors(img2, kp2)
    match_and_draw(img1, kp1, desc1_hog, img2, kp2, desc2_hog,
                   'results/1/uttower_match_hog.png', descriptor_type='HOG')
