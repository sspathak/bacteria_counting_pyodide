import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from tqdm import tqdm
from frontier import Frontier
# from my_threshold import custom_adaptive_threshold
from bacteria import Bacteria
import pdb
import os
from hyperparameters import *
#####################
# Utilities 

# get 8 symmetries of an image
def get_symmetries(img):
    symmetries = []
    for i in range(4):
        y = np.rot90(img, i)
        for flip in [False, True]:
            if flip:
                y = np.fliplr(y)

            symmetries.append(y)
    return symmetries

def get_feature_img(arr_of_imgs, arr_of_backgrounds):
    results = []
    for i in range(len(arr_of_imgs)):
        img = arr_of_imgs[i]
        bg = arr_of_backgrounds[i]
        r_img = img[:, :, 0]
        g_img = img[:, :, 1]
        b_img = img[:, :, 2]
        front_r = np.mean(r_img[r_img != 0])
        front_g = np.mean(g_img[g_img != 0])
        front_b = np.mean(b_img[b_img != 0])
        bact_color = np.array([front_r, front_g, front_b])
        scale = np.zeros_like(bact_color)
        for i, val in enumerate(bact_color - bg):
            if abs(val) >= 1e-4:
                scale[i] = 1/val

        # normalized_img = (img - bg) * scale
        normalized_img = (img - bg)
        normalized_img[img == 0] = 0
        results.append(normalized_img)
        # print(normalized_img[normalized_img!=0])
        # results.append(img) # Not using bg normalization
    return np.array(results)
    
# largest box to include both shapes
def max_box(shape1, shape2):
    return (max(shape1[0], shape2[0]), max(shape1[1], shape2[1]))

# Put elements in src to tar
def copy_pixel(src, tar):
    for i in range(len(src)):
        tar[i] = int(src[i])

def out_of_bound(img, x):
    return x < 0 or x >= len(img)
                                                               
# add a point to frontier and avoid duplicate points
def add_to_frontier(frontier, successors):
    for successor in successors:
        if not frontier.contains(successor):
            frontier.add(successor)

# To prepare a function that returns non-zero unvisited neighbors of a position on the image
def successor_init(img: np.ndarray):
    def getSuccessor(x:int, y:int, visited):
        successors = []
        for i, j in [[1,0], [0,1]]:
            for d in [1, -1]:
                xx = x + i * d
                yy = y + j * d
                if (xx < 0 or xx >= len(img) or yy < 0 or yy >= len(img[0]) or img[xx][yy] == 0 or visited[xx][yy]):
                    continue
                else:
                    successors.append([xx, yy])
        return successors
    return getSuccessor

def find_bacteria(x: int, y: int, visited: np.ndarray, img: np.ndarray) -> Bacteria:
    if img[x][y] == 0 or (visited[x][y]):
        visited[x][y] = True
        return None
    frontier = Frontier()
    frontier.add([x, y])
    bact = Bacteria()
    get_succ = successor_init(img)
    while not frontier.isEmpty():
        # pop a node
        node = frontier.pop()

        # use node to update rect
        xx, yy = node
        bact.add_coord(xx, yy)

        visited[xx][yy] = True       

        #get successors and push to frontier
        add_to_frontier(frontier, get_succ(xx, yy, visited)) 
    return bact
    
# pass thresholded img to make this work
def find_all_bact(thresh_img, orig_img, label, max_diameter = float('inf'), expansion_size = 3, size=[0, float('inf')], max_shape = None, image_name = "current image") -> List[Bacteria]:
    visited = np.full(thresh_img.shape, False)

    bacts = []

    pbar = tqdm(range(len(thresh_img)), total=len(thresh_img))
    for i in pbar:
        pbar.set_description('processing {}'.format(image_name))
        for j in range(len(thresh_img[i])):
            bact = find_bacteria(i, j, visited, thresh_img)
            visited[i][j] = True
            if bact is None or bact.size() < min(size) or bact.size() > max(size):
                continue
            if bact.est_diameter() >= max_diameter:
                continue
            bact_shape = bact.bounds.suggest_shape()
            if max_shape is not None and (bact_shape[0] > max_shape[0] or bact_shape[1] > max_shape[1]):
                continue
            bact.expand_bacteria(expansion_size, thresh_img)

            bact.retrieve_pixels(orig_img)
            bact.get_partial_img(orig_img)
            bact.get_bg_mean()
            bact.label = label
            bacts.append(bact)

    return bacts

def cover_upper_left(img, is_threshold, x_cover_range=600, y_cover_range=450):
    color = 0 if is_threshold else [0, 0, 0]
    vertices = np.array([[0,0], [x_cover_range, 0], [0, y_cover_range]])
    cv2.fillPoly(img, pts=[vertices], color=color)
    return img

def cover_upper_right(img, is_threshold, x_cover_range=120, y_cover_range=30):
    color = 0 if is_threshold else [0, 0, 0]
    vertices = np.array([[img.shape[1]-x_cover_range-1, 0], [img.shape[1]-1, y_cover_range], [img.shape[1]-1, 0]])
    cv2.fillPoly(img, pts=[vertices], color=color)
    return img

def roi(img, is_threshold=False):
    cover_upper_left(img, is_threshold)
    cover_upper_right(img, is_threshold)
    return img

def draw_bacteria(img, bact: Bacteria, color=[0, 255, 0], mode="draw"):
    if mode == "draw":
        for x, y in bact.coords:
            if bact.is_boundary(x, y):
                for xx, yy in bact.get_neighbors(x, y):
                    if xx < 0 or yy < 0 or xx >= len(img) or yy >= len(img[0]):
                        continue
                    if [xx, yy] not in bact.coords:
                        img[xx][yy] = color
    elif mode == "rectangle":
        gap = 2
        cv2.rectangle(img, (max(bact.bounds.left - gap, 0), max(bact.bounds.top - gap, 0)), (min(bact.bounds.right + 2, len(img[0])), min(bact.bounds.bottom + 2, len(img))), color, 1)
    elif mode == 'feature':
        for x, y in bact.expanded_coords:
            if bact.is_boundary(x, y, bact.expanded_coords):
                for xx, yy in bact.get_neighbors(x, y):
                    if xx < 0 or yy < 0 or xx >= len(img) or yy >= len(img[0]):
                        continue
                    if [xx, yy] not in bact.expanded_coords:
                        img[xx][yy] = color

    return img

def invert_img(img):
    new_img = np.ones((img.shape))
    new_img -= img
    return new_img.astype(np.uint8)


class BacteriaGenerator:
    def __init__(self, size_bounds, max_diameter, debug, cover_corners):
        self.size_bounds = size_bounds
        self.max_diameter = max_diameter
        self.debug = debug
        self.cover_corners = cover_corners
    
    def preprocess_v2(self, orig_img, C = C):
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        smoothed_image = img
        grad_x = cv2.Sobel(smoothed_image, cv2.CV_32F, 1, 0, ksize=1)
        grad_y = cv2.Sobel(smoothed_image, cv2.CV_32F, 0, 1, ksize=1)

        gradient_magnitude = np.maximum(np.abs(grad_x), np.abs(grad_y))
        # ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        grad_img = invert_img(gradient_magnitude)
        img = np.logical_and(grad_img <= 255-C, grad_img >= C).astype(np.uint8)
        return img

    def preprocess_v3(self, orig_img, C=C, debug_path=""):
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        smoothed_image = img
        grad_x = cv2.Sobel(smoothed_image, cv2.CV_32F, 1, 0, ksize=1)
        grad_y = cv2.Sobel(smoothed_image, cv2.CV_32F, 0, 1, ksize=1)
        gradient_magnitude = np.maximum(np.abs(grad_x), np.abs(grad_y))
        grad_img = invert_img(gradient_magnitude)
        grad_img[grad_img >= 255 - C] = 0
        grad_img[grad_img <= C] = 0

        grad_x_vis, grad_y_vis, quad_vis = self.sign_visualize(grad_x, grad_y, eps=C)
        if self.debug:
            cv2.imwrite(os.path.join(debug_path, 'grad_x_vis.png'), grad_x_vis)
            cv2.imwrite(os.path.join(debug_path, 'grad_y_vis.png'), grad_y_vis)
            cv2.imwrite(os.path.join(debug_path, 'quad_vis.png'), quad_vis)
        return grad_img
    
    def preprocess_v4(self, orig_img, C=C, debug_path=""):
        # Single sided preprocessing
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        smoothed_image = img
        grad_x = cv2.Sobel(smoothed_image, cv2.CV_32F, 1, 0, ksize=1)
        grad_y = cv2.Sobel(smoothed_image, cv2.CV_32F, 0, 1, ksize=1)
        grad_x[grad_x < 0] = 0
        grad_y[grad_y < 0] = 0
        gradient_magnitude = np.maximum(np.abs(grad_x), np.abs(grad_y))
        grad_img = invert_img(gradient_magnitude)
        grad_img[grad_img >= 255 - C] = 0
        grad_img[grad_img <= C] = 0

        grad_x_vis, grad_y_vis, quad_vis = self.sign_visualize(grad_x, grad_y, eps=C)
        if self.debug:
            cv2.imwrite(os.path.join(debug_path, 'grad_x_vis.png'), grad_x_vis)
            cv2.imwrite(os.path.join(debug_path, 'grad_y_vis.png'), grad_y_vis)
            cv2.imwrite(os.path.join(debug_path, 'quad_vis.png'), quad_vis)
        return grad_img
    
    def fill_1d_region(self, row, left_confirm_count = 2, max_grad_diameter = MAX_DIAMETER - 6, edge_shrink_length = 1):
        # determined filled region
        # Protocol: if left is not found but grad change occured, prepare to record pixels and keep going right
        # If left is found, right is not found and the grad sign is the same, still on the left region of the bacteria, keep going right
        # If left is found, right is not found and the grad is zero, we are likely in the middle region, go right for at most diameter steps. If no grad, stop and reset all parameters
        # If left is found, right is not found the grad is opposite sign, on the right side, document first right found
        # If first right is found and grad is 0, stop and reset all parameters
        filled_indices = []
        local_filled_indices = []
        leftFound = False
        firstRightFound = False
        left_grad_sign = 0
        last_left_index = None
        for j, val in enumerate(row):
            if not leftFound:
                future_vals = row[j: min(j + left_confirm_count, len(row))]
                grad_addition = sum(np.sign(future_vals))
                if abs(grad_addition) >= left_confirm_count:
                    # left first found, prepare to record
                    leftFound = True
                    left_grad_sign = np.sign(val)
                    firstRightFound = False
                    local_filled_indices = [j]
            elif leftFound and not firstRightFound:
                grad_sign = np.sign(val)
                if grad_sign == left_grad_sign:
                    # still on left side
                    local_filled_indices.append(j)
                elif grad_sign == 0:
                    # likely middle region, go on for a while
                    if last_left_index is None:
                        last_left_index = j-1
                    if j - last_left_index >= max_grad_diameter:
                        # Stop and reset
                        leftFound = False
                        firstRightFound = False
                        left_grad_sign = 0
                        local_filled_indices = []
                        last_left_index = None
                    else:
                        local_filled_indices.append(j)
                else: # grad sign is opposite
                    # right side found
                    firstRightFound = True
                    local_filled_indices.append(j)
            elif firstRightFound:
                grad_sign = np.sign(val)
                if grad_sign == 0:
                    # stop and reset
                    local_filled_indices = local_filled_indices[edge_shrink_length: -edge_shrink_length]
                    filled_indices += local_filled_indices
                    leftFound = False
                    firstRightFound = False
                    last_left_index = None
                    left_grad_sign = 0
                    local_filled_indices = []
                else:
                    # still on right side
                    local_filled_indices.append(j)
            elif j == len(row) - 1:
                # reach the end without finding right side
                filled_indices += local_filled_indices[edge_shrink_length:]

        reconstructed_row = np.zeros_like(row)
        reconstructed_row[filled_indices] = 1
        return reconstructed_row
    
    def preprocess_v5(self, orig_img, C=C, debug_path="", img_name=""):
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        smoothed_image = img
        # smoothed_image = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8, sigmaY=0.8)
        grad_x = cv2.Sobel(smoothed_image, cv2.CV_32F, 1, 0, ksize=1)
        grad_y = cv2.Sobel(smoothed_image, cv2.CV_32F, 0, 1, ksize=1)
        
        grad_x[np.abs(grad_x) <= C] = 0
        grad_y[np.abs(grad_y) <= C] = 0

        preprocessed_grad_x = grad_x.copy()
        for i, row in enumerate(grad_x):
            filled_row = self.fill_1d_region(row)
            preprocessed_grad_x[i] = filled_row

        preprocessed_grad_y = grad_y.copy()
        for i in range(grad_y.shape[1]):
            col = grad_y[:, i]
            filled_col = self.fill_1d_region(col)
            preprocessed_grad_y[:, i] = filled_col

        preprocessed = cv2.bitwise_and(preprocessed_grad_x, preprocessed_grad_y).astype(np.uint8)
    
        # grad_x_vis, grad_y_vis, quad_vis = self.sign_visualize(grad_x, grad_y, eps=C)
        # if self.debug:
        #     cv2.imwrite(os.path.join(debug_path, f'grad_x_vis_{img_name}'), grad_x_vis)
        #     cv2.imwrite(os.path.join(debug_path, f'grad_y_vis_{img_name}'), grad_y_vis)
        #     cv2.imwrite(os.path.join(debug_path, f'quad_vis_{img_name}'), quad_vis)
        return preprocessed
    
    def sign_visualize(self, grad_x, grad_y, eps=1e-6):
        H, W = grad_x.shape

        # --- 1) 单轴正负显示（红=正，蓝=负，灰=近0） ---
        def axis_sign_vis(g):
            pos = (g >  eps)
            neg = (g < -eps)
            zer = ~(pos | neg)

            vis = np.zeros((H, W, 3), np.uint8)
            vis[pos] = (0, 0, 255)        # 红：正
            vis[neg] = (255, 0, 0)        # 蓝：负
            vis[zer] = (128, 128, 128)    # 灰：零/极小
            return vis

        gx_vis = axis_sign_vis(grad_x)
        gy_vis = axis_sign_vis(grad_y)

        # --- 2) 象限合成：同时看 x/y 正负 ---
        # (++): 黄, (+-): 洋红, (-+): 青, (--): 绿, 0区：灰
        px = (grad_x >  eps); nx = (grad_x < -eps)
        py = (grad_y >  eps); ny = (grad_y < -eps)
        zer = ~(px | nx | py | ny)

        quad = np.zeros((H, W, 3), np.uint8)
        quad[ px &  py] = (0, 255, 255)   # 黄 (BGR)  x+, y+
        quad[ px &  ny] = (255, 0, 255)   # 洋红       x+, y-
        quad[ nx &  py] = (255, 255, 0)   # 青         x-, y+
        quad[ nx &  ny] = (0, 255, 0)     # 绿         x-, y-
        quad[zer]       = (128,128,128)   # 灰：近0

        return gx_vis, gy_vis, quad

    def generate_bacts(self, img, label, image_name = "current_image.bmp", debug_path = "for_debug"):
        if self.debug:
            debug_img = img.copy()
        processed_img = self.preprocess_v5(img, debug_path=debug_path, img_name=image_name)
        if self.cover_corners:
            processed_img = roi(processed_img, is_threshold=True)
        if self.debug:
            cv2.imwrite(os.path.join(debug_path, 'threshold_' + image_name), processed_img * 255)
        bacts = find_all_bact(processed_img, img, label, max_diameter = self.max_diameter, expansion_size=3, size=self.size_bounds, image_name=image_name)
        bact_count = 0
        max_shape = (1, 1)
        final_bacts = []
        for bact in bacts:
            if self.debug:
                draw_bacteria(debug_img, bact, [0, 255, 0], mode="draw")
                bact_count += 1

            final_bacts.append(bact)
            max_shape = max_box(max_shape, bact.img_shape())
        if self.debug:
            cv2.imwrite(os.path.join(debug_path, f"num_bacts_{len(bacts)}_" + image_name), debug_img)
        return final_bacts, max_shape
    
    