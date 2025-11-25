from bounds import Bounds
import numpy as np
import cv2
import math
# from my_threshold import *
import matplotlib.pyplot as plt


# Bateria object
class Bacteria:
    def __init__(self):
        self.coords = [] # coordinates of the bacteria
        self.expanded_coords = []
        self.x_coords = [] # X values of the coordinate
        self.y_coords = [] # y values of the coordinate
        self.img = None # img to store bacteria
        self.boundary = []
        self.bounds = Bounds() # box to bound bateria
        self.label = None 
        self.part_img = None
        self.bg_mean = None
        self.part_img_shape = (28, 28)

    def add_coord(self, x, y, expanding=False):
        # add to my lists
        self.coords.append([x, y])
        self.x_coords.append(x)
        self.y_coords.append(y)

        self.expanded_coords.append([x, y])
        # update bounds
        self.bounds.add((x, y))

    def reset_boundary(self):
        result = []
        for i, j in self.expanded_coords:
            if self.is_boundary(i, j, self.expanded_coords):
                result.append([i, j])
        self.boundary = result
        return result

    def get_pixel_mean(self):
        return self.pixel_mean
    
    def out_of_bounds(self, x, y):
        return x < 0 or y < 0
    
    def get_neighbors(self, x, y):
        neighbors = []
        for i, j in [[1, 0], [0, 1], [1, 1], [1, -1]]:
            for a in [1, -1]:
                xx = x + i * a
                yy = y + j * a
                if not self.out_of_bounds(xx, yy):
                    neighbors.append([xx, yy])
        return neighbors
    
    def expand_bacteria(self, ntimes, img):
        self.reset_boundary()
        # print(self.boundary)
        for _ in range(ntimes):
            for x, y in self.boundary.copy():
                for xx, yy in self.get_neighbors(x, y):
                    if [xx, yy] in self.expanded_coords or xx >= img.shape[0] or yy >= img.shape[1]:
                        continue
                    self.expanded_coords.append([xx, yy])
                    self.boundary.append([xx, yy])
                    self.bounds.add([xx, yy])
                self.boundary.remove([x, y])

    def retrieve_pixels(self, img):
        self.img = np.zeros((*(self.bounds.suggest_shape()), 3)).astype(np.uint8)
        self.pixels = []

        for x, y in self.expanded_coords:
            self.img[x-self.bounds.top][y-self.bounds.left][0] = img[x][y][0]
            self.img[x-self.bounds.top][y-self.bounds.left][1] = img[x][y][1]
            self.img[x-self.bounds.top][y-self.bounds.left][2] = img[x][y][2]           
    
    def pad_img(self, desired_shape):
        height, width = self.img_shape()
        desired_height, desired_width = desired_shape
        if desired_width < width or desired_height < height:
            print("Cannot perform padding with desired shape smaller or equal to {}".format(self.img_shape()))
            return -1
        hdiff, wdiff = abs(desired_height - height), abs(desired_width - width)
        self.img = cv2.copyMakeBorder(self.img, top = hdiff // 2, bottom = (hdiff + 1) // 2, left = wdiff // 2, right= (wdiff + 1)//2, borderType=cv2.BORDER_CONSTANT, value=0)
        return 1

    def get_partial_img(self, img:np.ndarray):
        xc, yc = self.get_center()
        x_size, y_size = self.part_img_shape
        t = max(xc - int(x_size // 2), 0)
        l = max(yc - int(y_size // 2), 0)
        b = min(xc + math.ceil(x_size / 2), len(img))
        r = min(yc + math.ceil(y_size / 2), len(img[0]))
        self.part_img = img[t:b+1, l:r+1]
        return self.part_img
    
    def bg_img(self, bias = 5):
        pimg = np.array(cv2.cvtColor(self.part_img, cv2.COLOR_BGR2GRAY))

        _, pimg = cv2.threshold(pimg, np.median(pimg.flatten()) - bias, 255, cv2.THRESH_BINARY)
        mask = (pimg/255).astype(np.uint8)
        bg_img = cv2.bitwise_and(self.part_img, self.part_img, mask=mask)
        return bg_img
    
    def get_bg_mean(self):
        if self.bg_mean is None:
            b_img = self.bg_img()

            pixel_sum = b_img.sum(0).sum(0)
            count = (b_img != 0).sum(0).sum(0)
            self.bg_mean = pixel_sum / count
        return self.bg_mean
    
    def get_bg_std(self):
        if self.bg_mean is None:
            b_img = self.bg_img()

            pixel_sum = b_img.sum(0).sum(0)
            count = (b_img != 0).sum(0).sum(0)
            self.bg_mean = pixel_sum / count
        return self.bg_mean
    
    # This image is mostly used as a feature to feed LeNet5
    def bg_normalized(self):
        # return ((self.img / self.get_bg_mean())).astype(np.float32)
        return (self.img.astype(np.uint8), self.get_bg_mean())
    
    def img_mean(self, img):
        g_img = np.mean(img, axis = 2)
        f_img = g_img.flatten()
        return f_img.sum() / (f_img != 0).sum()
    
    def img_std(self, img):
        g_img = np.mean(img, axis=2)
        return np.std(g_img.flatten()[g_img.flatten() != 0])
        
    def imshow(self):
        plt.imshow(self.img)
        plt.show()

    def size(self):
        return len(self.x_coords)
    
    def img_shape(self):
        if self.img is not None:
            return (self.img.shape[0], self.img.shape[1])
        else:
            return self.bounds.suggest_shape()
    
    def get_feature(self):
        return self.img

    def is_boundary(self, x, y, coords = None):
        if coords is None:
            coords = self.coords
        for xx, yy in self.get_neighbors(x, y):
            if [xx, yy] not in coords:
                return True
        return False
    
    def dist(self, pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    
    def get_end_pts(self):
        max_d = 0
        init_pt = self.coords[0]
        max_pt = init_pt
        for pt in self.coords:
            dist = self.dist(pt, init_pt)
            if dist > max_d:
                max_pt = pt
                max_d = dist
        
        max_d = 0
        init_pt = max_pt
        for pt in self.coords:
            dist = self.dist(pt, init_pt)
            if dist > max_d:
                max_pt = pt
                max_d = dist
        return init_pt, max_pt
    
    def get_center(self):
        pt1, pt2 = self.get_end_pts()
        return [int((pt1[0] + pt2[0])/2), int((pt1[1] + pt2[1])/2)]
    
    def est_diameter(self):
        return self.dist(*self.get_end_pts())