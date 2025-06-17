# stitching.py
import cv2
import numpy as np

def stitch_pair(img1, img2, kp1, kp2, matches, method='affine'):

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

        H = np.vstack([M, [0,0,1]])
    else:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    all_corners = np.concatenate((warped_corners, np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]

    T = np.array([[1, 0, t[0]],
                  [0, 1, t[1]],
                  [0, 0, 1]])
    result = cv2.warpPerspective(img1, T.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h2+t[1], t[0]:w2+t[0]] = img2
    return result

if __name__ == '__main__':

    img1 = cv2.imread('results/1/uttower1_keypoints.jpg')
    img2 = cv2.imread('results/1/uttower2_keypoints.jpg')
    gray1 = cv2.cvtColor(cv2.imread('images/1/uttower1.jpg'), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cv2.imread('images/1/uttower2.jpg'), cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(bf.match(desc1, desc2), key=lambda x: x.distance)[:100]

    pano_sift = stitch_pair(cv2.imread('images/1/uttower1.jpg'),
                            cv2.imread('images/1/uttower2.jpg'),
                            kp1, kp2, matches, method='affine')
    cv2.imwrite('results/1/uttower_stitching_sift.png', pano_sift)