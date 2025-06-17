import os
import numpy as np
from skimage import io, color
from skimage.segmentation import mark_boundaries

class SLIC:
    def __init__(self, image, num_segments=100, n_iter=10, compactness=20):
        self.image = image
        self.num_segments = num_segments
        self.n_iter = n_iter
        self.compactness = compactness
        self.height, self.width = image.shape[:2]
        self.step = int(np.sqrt(self.height * self.width / self.num_segments))
        self.lab = color.rgb2lab(self.image)
        self.clusters = None
        self.labels = -np.ones((self.height, self.width), dtype=int)

    def initialize_clusters(self):
        centers = []
        offset = self.step // 2
        for i in range(offset, self.height, self.step):
            for j in range(offset, self.width, self.step):
                L, a, b = self.lab[i, j]
                centers.append([i, j, L, a, b])
        self.clusters = np.array(centers, dtype=float)

#为每个像素分配最近的聚类中心，生成labels矩阵
    def assign_labels(self):
        sc = self.compactness / self.step
        r = 2 * self.step
        for idx, (ci, cj, Lc, ac, bc) in enumerate(self.clusters):
            ci, cj = int(ci), int(cj)
            x0, x1 = max(ci - r, 0), min(ci + r, self.height)
            y0, y1 = max(cj - r, 0), min(cj + r, self.width)
            sub_lab = self.lab[x0:x1, y0:y1]
            xs, ys = np.ogrid[x0:x1, y0:y1]
            dc = np.sqrt((sub_lab[:,:,0] - Lc)**2 +
                         (sub_lab[:,:,1] - ac)**2 +
                         (sub_lab[:,:,2] - bc)**2)
            ds = np.sqrt((xs - ci)**2 + (ys - cj)**2)
            D = np.sqrt(dc**2 + (sc * ds)**2)
            mask = D < np.inf  
            if idx == 0:
                self.minDist = np.full((self.height, self.width), np.inf)
            region = self.minDist[x0:x1, y0:y1]
            update = D < region
            self.labels[x0:x1, y0:y1][update] = idx
            self.minDist[x0:x1, y0:y1][update] = D[update]

    def update_clusters(self):
        new_centers = np.zeros_like(self.clusters)
        counts = np.zeros(len(self.clusters))
        for i in range(self.height):
            for j in range(self.width):
                idx = self.labels[i, j]
                if idx >= 0:
                    new_centers[idx, 0] += i
                    new_centers[idx, 1] += j
                    new_centers[idx, 2:] += self.lab[i, j]
                    counts[idx] += 1
        mask = counts > 0
        self.clusters[mask, :2] = new_centers[mask, :2] / counts[mask, None]
        self.clusters[mask, 2:] = new_centers[mask, 2:] / counts[mask, None]

    def iterate(self):
        self.initialize_clusters()
        for _ in range(self.n_iter):
            self.minDist = None
            self.assign_labels()
            self.update_clusters()

    def get_segmentation(self):
        return mark_boundaries(self.image, self.labels + 1, color=(1, 0, 0))

if __name__ == '__main__':
    img = io.imread('data/woman.png')
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img / 255.0

    out_dir = 'results/superpixels/SLIC'
    os.makedirs(out_dir, exist_ok=True)

    for it in [5, 10, 20]:
        slic = SLIC(img, num_segments=200, n_iter=it, compactness=20)
        slic.iterate()
        seg = slic.get_segmentation()
        io.imsave(os.path.join(out_dir, f'woman_iter{it:02d}.jpg'), (seg * 255).astype(np.uint8))

    for num in [100, 200, 400]:
        slic = SLIC(img, num_segments=num, n_iter=10, compactness=20)
        slic.iterate()
        seg = slic.get_segmentation()
        io.imsave(os.path.join(out_dir, f'woman_num_{num:03d}.jpg'), (seg * 255).astype(np.uint8))
