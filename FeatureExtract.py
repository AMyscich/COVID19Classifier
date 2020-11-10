from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import signal

# (b.i) Visual feature extraction
class FeatureExtract:
    def __init__(self):
        pass

    # Smooth image with Gaussian kernel (padding so input dim == output dim)
    def smooth_image(self, x, sd=1.0):
        kernel_size = np.ceil(6.0*sd)
        kernel_1d = signal.gaussian(kernel_size, sd)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        normalizing_factor = np.sum(kernel_2d)
        gaussian_kernel_2d = kernel_2d/normalizing_factor
        image_smoothed = signal.convolve2d(x,gaussian_kernel_2d,mode='same')
        return image_smoothed

    # Grayscale edge detection using Sobel filter (no padding, i.e. 'valid')
    # Returns g (gradient), gx (x-gradient), gy (y-gradient)
    def edge_detection(self, image, use_padding=False):
        # Not used, just for reference
        difference_filter = np.asarray([1,0,-1])
        average_filter = np.asarray([1,2,1])

        n_rows = image.shape[0]
        n_cols = image.shape[1]

        # -2 for valid
        gradient = np.zeros((n_rows-2,n_cols-2))
        gx = np.zeros((n_rows-2,n_cols-2))
        gy = np.zeros((n_rows-2,n_cols-2))
        diff_conv = np.zeros((n_rows,n_cols-2))
        avg_conv = np.zeros((n_rows,n_cols-2))

        for c in range(2, n_cols):
            diff_conv[:,c-2] = image[:,c] - image[:,c-2]
            avg_conv[:,c-2] = image[:,c] + 2*image[:,c-1] + image[:,c-2]

        for r in range(2, n_rows):
            gx[r-2,:] = diff_conv[r,:] + 2*diff_conv[r-1,:] + diff_conv[r-2,:]
            gy[r-2,:] = avg_conv[r,:] - avg_conv[r-2,:]

        for r in range(0, n_rows-2):
            for c in range(0, n_cols-2):
                gradient[r,c] = np.sqrt(gx[r,c]**2+gy[r,c]**2)

        return (gradient, gx, gy)




# # Example Usage
# x = cv2.imread('sandbox/tamu.jpeg', cv2.IMREAD_GRAYSCALE)
# plt.imshow(x,cmap='gray',vmin=0,vmax=255)
# plt.show()

# fe = FeatureExtract()

# # Example on smooth_image() usage:
# s = fe.smooth_image(x,sd=3.0)
# plt.imshow(s,cmap='gray',vmin=0,vmax=255)
# plt.show()

# # Example on edge_detection() usage:
# g, gx, gy = fe.edge_detection(x)
# plt.imshow(g,cmap='gray',vmin=0,vmax=255)
# plt.show()
# plt.imshow(gx,cmap='gray',vmin=0,vmax=255)
# plt.show()
# plt.imshow(gy,cmap='gray',vmin=0,vmax=255)
# plt.show()

# # Smoothing gradient image:
# g_smooth = fe.smooth_image(g,sd=3.0)
# plt.imshow(g_smooth,cmap='gray',vmin=0,vmax=255)
# plt.show()
