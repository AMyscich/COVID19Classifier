from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import signal
from skimage import feature

# (b.i) Visual feature extraction
class FeatureExtract:
    def __init__(self):
        pass

    # Might want to do no padding, since there's artifacting
    # Smooth image with Gaussian kernel (no padding, i.e. 'valid')
    # Width of kernel is ceiling of 6.0*sd
    def smooth_image(self, image, sd=1.0):
        kernel_size = np.ceil(6.0*sd)
        kernel_1d = signal.gaussian(kernel_size, sd)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        normalizing_factor = np.sum(kernel_2d)
        gaussian_kernel_2d = kernel_2d/normalizing_factor
        image_smoothed = signal.convolve2d(image,gaussian_kernel_2d,mode='valid')
        return image_smoothed

    # Invert image intensities
    def invert_image(self, image):
        n_rows = image.shape[0]
        n_cols = image.shape[1]
        image_ivt = np.zeros((n_rows,n_cols))
        image_max = np.amax(image)
        image_min = np.amin(image)
        image_ivt.fill(float(image_max+image_min))
        return image_ivt-image

    # Truncate image if intensity at or below a certain threshold
    def truncate_image(self, image, threshold=0.0):
        return np.where(image <= threshold, 0.0, image)

    # Normalize an image (can be linear or non-linear, i.e. 'logistic')
    # Normalizes with newMin and newMax based on passed in normalization range. Default is 0-255
    # k is how 'aggressive' the logistic activation is, lower means it'll be more of a gradient (~0.1), higher is a hard cutoff (~2)
    def normalize_image(self, image, type='linear', new_min=0.0, new_max=255.0, k=1):
        n_rows = image.shape[0]
        n_cols = image.shape[1]
        image_norm = np.zeros((n_rows,n_cols))
        current_min = np.amin(image)
        current_max = np.amax(image)
        if type == 'linear':
            image_norm = ((image-current_min)*((new_max-new_min)/(current_max-current_min)))+new_min
        elif type == 'nonlinear':
            midpoint = (current_min+current_max)/2
            activation = 1/(1+np.exp(-1*k*(image-midpoint)))
            image_norm = (new_max-new_min)*activation + new_min
        else:
            image_norm = image

        return image_norm

    # Equalize image using histogram method (histogram equalization)
    def equalize_image(self, image):
        pixel_hist = np.zeros(256) # 0-255 => 0
        n_rows = image.shape[0]
        n_cols = image.shape[1]

        for r in range(n_rows):
            for c in range(n_cols):
                pixel_integer = int(image[r,c])
                pixel_hist[pixel_integer] += 1

        hist_sum = np.sum(pixel_hist)
        pixel_hist /= hist_sum # Normalize
        cdf = np.cumsum(pixel_hist) # Cumulative Sum
        image_eq = np.zeros((n_rows,n_cols))
        for r in range(n_rows):
            for c in range(n_cols):
                pixel_integer = int(image[r,c])
                # There are two formulas for this. The first is much simpler, the second I've used before.
                # I don't notice a difference in results. The latter uses 'clamping'
                #image_eq[r,c] = np.floor(255.0*cdf[pixel_integer])
                image_eq[r,c] = min(max(255.0*(cdf[pixel_integer]-cdf[0])/(1.0-cdf[0]),0.0), 255.0)

        return image_eq

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

    # TODO: Improve
    # Return blob coordinates and std using LoG method
    def blob_detection(self, image, min_sigma=1, max_sigma=50, num_sigma=10):
        blobs = feature.blob_log(image,min_sigma=min_sigma,max_sigma=max_sigma,num_sigma=num_sigma)
        return blobs

def dirty_plot(image_og, image_new, title):
    plt.subplot(1, 2, 2)
    plt.imshow(image_og,cmap='gray',vmin=0,vmax=255)
    plt.title('Original Image')
    plt.subplot(1, 2, 1)
    plt.imshow(image_new,cmap='gray',vmin=0,vmax=255)
    plt.title(title)
    plt.show()

# # Pixels in range 0(black)-255(white)

# Sandbox image
x = cv2.imread('sandbox/tamu.jpeg', cv2.IMREAD_GRAYSCALE)

# Demo x-ray image
# x = cv2.imread('data/train/img_0.jpeg', cv2.IMREAD_GRAYSCALE)

# Manual image construction
# n_rows = 256
# n_cols = 256
# min_val = 20
# max_val = 200
# x = np.zeros((n_rows,n_cols))

# Manual gradient image from top to bottom
# for r in range(n_rows):
#     x[r:,] = r
#     if r < min_val:
#         x[r,:] = min_val
#     if r > max_val:
#         x[r,:] = max_val

# Horizontal line image
# for r in range(n_rows):
#     if r%4 == 0 or r%4==1:
#         x[r,:] = max_val
#     else:
#         x[r,:] = min_val

# Vertical line image
# for c in range(n_cols):
#     if r%4 == 0 or r%4==1:
#         x[:,c] = max_val
#     else:
#         x[:,c] = min_val

# Diagonal lines image
# for r in range(n_rows):
#     for c in range(n_cols):
#         if (r < c+2 and r > c-2) or (r+c < n_rows+2 and r+c > n_rows-2):
#             x[r,c] = max_val
#         else:
#             x[r,c] = min_val

plt.imshow(x,cmap='gray',vmin=0,vmax=255)
plt.title('Base Image')
plt.show()
fe = FeatureExtract()

# # Example on truncate image
trun = fe.truncate_image(x, 122.0)
dirty_plot(x, trun, 'Truncate')

# # Example on invert image
inv = fe.invert_image(x)
dirty_plot(x, inv, 'Invert')

# Example on normalize image (linear) (transforms to range 0-255)
norm_l = fe.normalize_image(x, type='linear',new_min=0.0,new_max=255.0)
dirty_plot(x, norm_l, 'Linear Normalization')

# Example on normalize image (nonlinear)
norm_n = fe.normalize_image(x, type='nonlinear',new_min=0.0,new_max=255.0, k=0.1)
dirty_plot(x, norm_n, 'Nonlinear Normalization')

# # Example on equalize image
e = fe.equalize_image(x)
dirty_plot(x, e, 'Image Equalization')

# Example on smooth_image() usage:
s = fe.smooth_image(e,sd=1.0)
dirty_plot(x, s, 'Image Smoothing')

# Example on blob_detection() usage (run on smoothed version of equalized image):
# blobs = fe.blob_detection(s)
# print(blobs)

# # Example on edge_detection() usage:
g, gx, gy = fe.edge_detection(x)
dirty_plot(x, g, 'Gradient')
dirty_plot(x, gx, 'X-Gradient')
dirty_plot(x, gy, 'Y-Gradient')

# # Smoothing image, then edge detection of smoothed image
# # (Edge detection very sensitive to noise, so we usually smooth beforehand)
g, gx, gy = fe.edge_detection(s)
dirty_plot(x, g, 'Gradient (Smoothed)')
dirty_plot(x, gx, 'X-Gradient (Smoothed)')
dirty_plot(x, gy, 'Y-Gradient (Smoothed)')
