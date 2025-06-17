import numpy as np
from skimage import io, segmentation, color
from skimage.util import img_as_float
import cv2
import os
import csv

class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int32)
        self.int_diff = np.zeros(size, dtype=np.float32)
    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]] 
            u = self.parent[u]
        return u   
    def union(self, u, v, weight):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u == root_v:
            return
        if self.size[root_u] < self.size[root_v]:
            root_u, root_v = root_v, root_u
        self.parent[root_v] = root_u
        self.size[root_u] += self.size[root_v]
        self.int_diff[root_u] = weight

#预处理：
#构建8邻域边，计算颜色差异
def build_edges(image):
    height, width, _ = image.shape
    edges = []
    for i in range(height):
        for j in range(width):
            if j < width-1:  # 右
                u = i * width + j
                v = i * width + (j+1)
                diff = np.linalg.norm(image[i,j] - image[i,j+1])
                edges.append((diff, u, v))
            if i < height-1:  # 下
                u = i * width + j
                v = (i+1) * width + j
                diff = np.linalg.norm(image[i,j] - image[i+1,j])
                edges.append((diff, u, v))
            if i < height-1 and j < width-1:  # 右下
                u = i * width + j
                v = (i+1) * width + (j+1)
                diff = np.linalg.norm(image[i,j] - image[i+1,j+1])
                edges.append((diff, u, v))
            if i < height-1 and j > 0:  # 左下
                u = i * width + j
                v = (i+1) * width + (j-1)
                diff = np.linalg.norm(image[i,j] - image[i+1,j-1])
                edges.append((diff, u, v))
    edges.sort(key=lambda x: x[0])
    return edges    

#GBS算法实现函数：
def GBS(image, sigma, k, min_size):
    #高斯滤波
    smoothed = cv2.GaussianBlur(image, (0,0), sigmaX=sigma, sigmaY=sigma)
    smoothed = img_as_float(smoothed)
    height, width, _ = smoothed.shape
    num_pixels = height * width
    
    #初始化并查集
    uf = UnionFind(num_pixels)
    
    #构建&处理边
    edges = build_edges(smoothed)
    for weight, u, v in edges:
        root_u = uf.find(u)
        root_v = uf.find(v)
        if root_u != root_v:
            threshold = min(uf.int_diff[root_u] + k/uf.size[root_u],
                           uf.int_diff[root_v] + k/uf.size[root_v])
            if weight <= threshold:
                uf.union(u, v, weight)
    
    #后处理小区域(生成mask)
    mask = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            mask[i,j] = uf.find(i*width + j)
    
    #合并小区域
    unique, counts = np.unique(mask, return_counts=True)
    for label, count in zip(unique, counts):
        if count < min_size:
            indices = np.argwhere(mask == label)
            for pt in indices:
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = pt[0]+dx, pt[1]+dy
                        if 0<=nx<height and 0<=ny<width and mask[nx,ny] != label:
                            uf.union(pt[0]*width+pt[1], nx*width+ny, 0)
                            mask = np.where(mask == label, mask[nx,ny], mask)
                            break
    for i in range(height):
        for j in range(width):
            mask[i,j] = uf.find(i*width + j)
    return mask

#可视化：
def visualize_boundaries(image, mask):
    return segmentation.mark_boundaries(image, mask, color=(1,0,1))

#主函数：
#这里添加了自动统计分割数目和平均超像素面积的脚本，使各个参数的输出对比更加直观
def main():
    sigmas = [0.3, 0.5, 0.8]
    ks = [200, 500, 1000]
    min_sizes = [50, 100]

    img = io.imread('data/train.png')[..., :3]
    img = img_as_float(img)

    out_dir = 'results/superpixels/GBS'
    os.makedirs(out_dir, exist_ok=True)

    stats_file = os.path.join(out_dir, 'stats.csv')
    with open(stats_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sigma', 'k', 'min_size', '分割数目', '平均面积(像素)'])

        for sigma in sigmas:
            for k in ks:
                for min_size in min_sizes:
                    mask = GBS(img, sigma, k, min_size)
                    labels, counts = np.unique(mask, return_counts=True)
                    n_seg = len(labels)
                    avg_area = counts.mean()
                    writer.writerow([sigma, k, min_size, n_seg, f"{avg_area:.2f}"])
                    vis = visualize_boundaries(img, mask)
                    vis_uint8 = (vis * 255).astype(np.uint8)
                    fn = f'train_{sigma}_{k}_{min_size}.jpg'
                    io.imsave(os.path.join(out_dir, fn), vis_uint8)
    print(f"Segmentation and stats saved in {out_dir}.")

if __name__ == '__main__':
    main()
