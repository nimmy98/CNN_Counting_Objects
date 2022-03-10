# File to generate images
import numpy as np
from skimage.morphology import disk, diamond, square
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import sys


SHAPES = [square]


def generate_image(img_size, shapes, allow_occlusion=2):
    img = np.zeros((img_size, img_size), bool)
    record = [[0]*30]*30 # Blank matrix to ensure no repeats
    i = 0
    while i < len(shapes):
        shape, radius = shapes[i]
        shape = SHAPES[shape]
        shape = shape(radius, bool)
        shape_area = np.sum(shape)
        x, y = np.random.randint(radius, img_size-radius, 2)
        xi, xf = x-radius, x+radius
        yi, yf = y-radius, y+radius
        ori = img[xi:xf, yi:yf]
        res = np.logical_or(ori, shape)
        if np.sum(res)-np.sum(ori) >= shape_area - allow_occlusion:
            passc = 0
            if record[x][y] == 0:# Checks for any overlap on already placed objects
                if (x-radius > 0):
                    passc = passc + 1
                if x+radius < img_size - 1:
                    passc = passc + 1
                if y-radius > 0:
                    passc = passc + 1
                if y+radius < img_size - 1:
                    passc = passc + 1
                if passc == 4:
                    if record[x-radius][y] == 0 and record[x+radius][y] == 0 and record[x][y-radius] == 0 and record[x][y+radius] == 0:
                            img[xi:xf, yi:yf] = res # Places the new objects
                            record[x][y] = 1
                            record[x-radius][y] = 1
                            record[x+radius][y] = 1
                            record[x][y-radius] = 1
                            record[x][y+radius] = 1
                            i += 1# Only iterates if there was a success
                            print("success")
            
    return img

if __name__ == '__main__':
    NMIN, NMAX = 1, 6 # nbr of shapes
    RMAX = 1
    
    counter = 1501
    while(counter <= 2000):
        n = int(np.random.randint(NMIN, NMAX+1))# Using only 1 shape and size, could be changed here
        shapes = [(np.random.randint(len(SHAPES)),RMAX)
                  for i in range(n)]

        img = generate_image(30, shapes)
        plt.imshow(img,cmap= plt.cm.gray)
        plt.axis('off')
        # plt.title('total counted shape = %d' % n) Removing titles and labels to not distract the cnn
        namer = ''# Generates a file name where the objects in the image is in the first 2 digits of the file
        if (n < 10):
            namer += '0'
            namer += str(n - 1)
        else:
            namer += str(n - 1)
        namer += '_'
        namer += str(counter)
        namer += '.png'
        plt.savefig(namer, dpi=15, bbox_inches='tight')# Low DPI to help CNN
        counter += 1

