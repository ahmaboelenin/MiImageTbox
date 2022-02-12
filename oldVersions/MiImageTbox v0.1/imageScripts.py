import cv2.cv2 as cv2
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

'''____Essential_Functions___________________________________________________________________________________________'''
def image_read(path):
    img = cv2.imread(path)
    if img is None:
        return
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # Convert The Image From BGR To RGB
    return img


def image_show(name, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, img)
    cv2.waitKey(0)


def image_save(img, name):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, img)


def image_resize(img, method, x, y):  # Image Resize
    if method is None:  # Resize With Width and Height
        return cv2.resize(img, (x, y))
    elif method == "ratio":  # Resize With Ratio Of The Original Width and Height
        if x == 0:
            x = 1
        if y == 0:
            y = 1
        return cv2.resize(img, (0, 0), None, x, y)


def bgr2rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb2bgr(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


'''____Image_Transformation_Functions________________________________________________________________________________'''
def image_rotate(img, angle):
    height, width = img.shape[:2]
    image_center = (width / 2, height / 2)

    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # Find the New Width and Height Bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # Subtract Old Image Center (bringing image back to original) and Adding the New Image Center Coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))

    '''img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1])'''
    return rotated_img


def image_flip(img, case):
    """The function cv::flip flips a 2D array around vertical, horizontal, or both axes.
            1 - Vertical Flip (Around y-axis),              0 - Horizontal Flip (Around x-axis),
            -1 - Vertical-Horizontal Flip (Around xy-axis)   """
    return cv2.flip(img, case)


def image_translate(img, shift_x=0, shift_y=0):
    mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))


def image_un_translate(img):
    x_start, y_start, x_end, y_end = -1, -1, 0, 0

    for y in range(img.shape[0]):
        if y_start == -1:
            if not np.all(img[y, 0:] == 0):
                y_start = y
        else:
            if np.all(img[y, 0:] == 0):
                y_end = y
                break
    if y_end == 0:
        y_end = img.shape[0]

    for x in range(img.shape[1]):
        if x_start == -1:
            if not np.all(img[0:, x] == 0):
                x_start = x
            continue
        else:
            if np.all(img[0:, x] == 0):
                x_end = x
                break
    if x_end == 0:
        x_end = img.shape[1]

    a = img[y_start: y_end, x_start: x_end]
    if np.all(a == 0):
        return
    a = cv2.resize(a, (img.shape[1], img.shape[0]))
    return a


def image_skewing(img):
    pts1 = np.float32([[0, 0], [img.shape[1] - 1, 0], [100, img.shape[0] - 1]])
    pts2 = np.float32([[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1]])
    sm = cv2.getAffineTransform(pts1, pts2)
    skewed_img = cv2.warpAffine(img, sm, img.shape[1::-1])
    return skewed_img


def image_un_skewing(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img, 127, 255, 0)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # Get Contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    imx = img.shape[0]
    imy = img.shape[1]
    lp_area = (imx * imy) / 10

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(cnt) > lp_area:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            for i in range(len(box)):
                if box[i, 0] > img.shape[1] or box[i, 0] < 0:
                    if box[i, 0] > img.shape[1]:
                        box[i, 0] = img.shape[1]
                    if box[i, 0] < 0:
                        box[i, 0] = 0
            for i in range(len(box)):
                if box[i, 1] > img.shape[0] or box[i, 1] < 0:
                    if box[i, 1] > img.shape[0]:
                        box[i, 1] = img.shape[0]
                    if box[i, 1] < 0:
                        box[i, 1] = 0

            (tl, tr, br, bl) = box

            max_width = max(int(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))),
                            int(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))))
            max_height = max(int(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))),
                             int(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))))

            dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
                           dtype="float32")
            mat = cv2.getPerspectiveTransform(box.astype("float32"), dst)
            return cv2.warpPerspective(img, mat, (max_width, max_height))


def image_zoom(img, x, y):
    y_max, x_max = int(img.shape[0] / 4), int(img.shape[1] / 4)
    y_start, y_end, x_start, x_end = y - y_max, y + y_max, x - x_max, x + x_max

    if y_start < 0:
        y_start = 0
    if y_end > img.shape[0]:
        y_end = img.shape[0]

    if x_start < 0:
        x_start = 0
    if x_end > img.shape[1]:
        x_end = img.shape[1]

    zoomed_img = img[y_start: y_end, x_start: x_end]
    if np.all(zoomed_img == 0):
        return
    zoomed_img = cv2.resize(zoomed_img, (img.shape[1], img.shape[0]))
    return zoomed_img


def image_crop(img, x_start, y_start, x_end, y_end):
    if x_start > x_end:
        x_start, x_end = x_end, x_start
        if x_start < 0:
            x_start = 0
        if x_end > img.shape[1]:
            x_end = img.shape[1]
    if y_start > y_end:
        y_start, y_end = y_end, y_start
        if y_start < 0:
            y_start = 0
        if y_end > img.shape[0]:
            y_end = img.shape[0]

    cropped_img = img[y_start: y_end, x_start: x_end]
    cropped_img = cv2.resize(cropped_img, (img.shape[1], img.shape[0]))
    return cropped_img


def image_four_points_transform(img, rect):
    (tl, tr, br, bl) = rect
    max_width = max(int(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))),
                    int(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))))
    max_height = max(int(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))),
                     int(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))))

    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    data = np.array(rect, dtype=np.float32)
    mat = cv2.getPerspectiveTransform(data, dst)
    result = cv2.warpPerspective(img, mat, (max_width, max_height))
    result = cv2.resize(result, (img.shape[1], img.shape[0]))
    return result


'''____Image_Enhancement_Functions____'''
def image_blending(img, alpha, img2, beta):
    img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, alpha, img2, beta, 0.0)


def image_canny_edge(img, alpha, beta):
    return cv2.Canny(img, alpha, beta)


def image_addweighted(img, alpha, img2, beta):
    return cv2.addWeighted(img, alpha, img2, 0, beta)


def rgb_change(img, r, g, b):
    fused_img = cv2.addWeighted(img, 1, img, 0, 0)
    if r != 0:
        red_img = np.full((img.shape[0], img.shape[1], 3), (r, 0, 0), np.uint8)
        fused_img = cv2.addWeighted(img, 0.8, red_img, 0.2, 0)
    if g != 0:
        green_img = np.full((img.shape[0], img.shape[1], 3), (0, g, 0), np.uint8)
        fused_img = cv2.addWeighted(fused_img, 0.8, green_img, 0.2, 0)
    if b != 0:
        blue_img = np.full((img.shape[0], img.shape[1], 3), (0, 0, b), np.uint8)
        fused_img = cv2.addWeighted(fused_img, 0.8, blue_img, 0.2, 0)
    return fused_img


def adjust_gamma(img, gamma=0):
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gamma == 0:
        mid, mean = 0.5, np.mean(gray)
        gamma = math.log(mid * 255) / math.log(mean)
    return np.power(img, gamma).clip(0, 255).astype(np.uint8)


def thresh(img, mode, value):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == 'binary':
        # threshed_img = cv2.adaptiveThreshold(img, value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, value, 1)
        ret, threshed_img = cv2.threshold(img, value, 255, cv2.THRESH_BINARY)
    elif mode == 'binary_inv':
        ret, threshed_img = cv2.threshold(img, value, 255, cv2.THRESH_BINARY_INV)
    elif mode == 'to_zero':
        ret, threshed_img = cv2.threshold(img, value, 255, cv2.THRESH_TOZERO)
    elif mode == 'to_zero_inv':
        ret, threshed_img = cv2.threshold(img, value, 255, cv2.THRESH_TOZERO_INV)
    elif mode == 'trunc':
        ret, threshed_img = cv2.threshold(img, value, 255, cv2.THRESH_TRUNC)
    return threshed_img


def image_filters(img, filter_name):
    if filter_name == "salt_pepper":                                                        # Salt & Pepper Noise
        row, col = img.shape[:2]

        number_of_pixels = random.randint(300, 10000)  # Pick a random number between 300 and 10000
        for i in range(number_of_pixels):
            y_coord = random.randint(0, row - 1)  # Pick a random y coordinate
            x_coord = random.randint(0, col - 1)  # Pick a random x coordinate
            img[y_coord][x_coord] = 255  # Color that pixel to white

        number_of_pixels = random.randint(300, 10000)  # Pick a random number between 300 and 10000
        for i in range(number_of_pixels):
            y_coord = random.randint(0, row - 1)  # Pick a random y coordinate
            x_coord = random.randint(0, col - 1)  # Pick a random x coordinate
            img[y_coord][x_coord] = 0  # Color that pixel to black
        return img

    elif filter_name == "negative":                                                             # Negative Filter
        return cv2.bitwise_not(img)

    elif filter_name == "black_white":                                                          # Black And White Image
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, black_and_white_image) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return black_and_white_image

    elif filter_name == "border":                                                               # Black Border
        black = [0, 0, 0]
        border_img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        border_img = cv2.resize(border_img, (img.shape[1], img.shape[0]))
        return border_img

    elif filter_name == "blur":                                             # Blurring (Traditional / Average Filter)
        kernel = np.ones([3, 3]) / 9
        # return cv2.blur(src=img, ksize=(3, 3))

    elif filter_name == "median":                                                               # Median Blurring
        return cv2.medianBlur(src=img, ksize=3)

    elif filter_name == "gaussian":                                                             # Gaussian Filter
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        # return cv2.GaussianBlur(img, (5, 5), sigmaX=0)

    elif filter_name == "pyramidal":                                                            # Pyramidal Filter
        kernel = np.array([[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3], [2, 4, 6, 4, 2], [1, 2, 3, 2, 1]]) / 81

    elif filter_name == "circular":                                                             # Circular Filter
        kernel = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]]) / 21

    elif filter_name == "cone":                                                                 # Cone Filter
        kernel = np.array([[0, 0, 1, 0, 0], [0, 2, 2, 2, 0], [1, 2, 5, 2, 1], [0, 2, 2, 2, 0], [0, 0, 1, 0, 0]]) / 25

    elif filter_name == "bilateral":                                                            # Bilateral Filter
        return cv2.bilateralFilter(src=img, d=9, sigmaColor=75, sigmaSpace=75)

    elif filter_name == "emboss":                                                               # Emboss Filter
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

    elif filter_name == "sharpen":                                                              # Sharpening Filter
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])

    elif filter_name == "laplace_edge":                                                         # Laplace Edge Detection
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

    elif filter_name == "outline_edge":                                                         # Outline Edge Detection
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    elif filter_name == "sobelY" or filter_name == "sobelX" or filter_name == "sobelYX":         # Sobel Filter
        # kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if filter_name == "sobelY":
            sobel_img = cv2.Sobel(img, cv2.CV_8UC1, 0, 1, ksize=3)
        elif filter_name == "sobelX":
            sobel_img = cv2.Sobel(img, cv2.CV_8UC1, 1, 0, ksize=3)
        else:
            sobely_img = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
            sobelx_img = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
            sobel_img = sobely_img + sobelx_img

        sobel_img = cv2.convertScaleAbs(sobel_img)
        return sobel_img

    elif filter_name == "sobelY_64f" or filter_name == "sobelX_64f" or filter_name == "sobelYX_64f":
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if filter_name == "sobelY_64f":
            sobel_64 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        elif filter_name == "sobelX_64f":
            sobel_64 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        else:
            sobely_64 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            sobelx_64 = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
            sobel_64 = sobely_64 + sobelx_64

        abs_64 = np.absolute(sobel_64)
        sobel_8u = np.uint8(abs_64)
        return sobel_8u

    elif filter_name == 'scharrY' or filter_name == 'scharrX' or filter_name == 'scharrYX':     # Scharr Operator
        # This Operator Tries to Achieve the Perfect Rotational Symmetry
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if filter_name == 'scharrY':
            scharr_img = cv2.Scharr(img, cv2.CV_8UC1, 0, 1)
        elif filter_name == 'scharrX':
            scharr_img = cv2.Scharr(img, cv2.CV_8UC1, 1, 0)
        else:
            scharry_img = cv2.Scharr(img, cv2.CV_8UC1, 0, 1)
            scharrx_img = cv2.Scharr(img, cv2.CV_8UC1, 1, 0)
            scharr_img = scharry_img + scharrx_img
        return scharr_img

    elif filter_name == 'prewittY' or filter_name == 'prewittX' or filter_name == 'prewittYX':      # Prewitt Operator
        """Prewitt Operator - It is a gradient-based operator. It is one of the best ways to detect the orientation and 
        magnitude of an image. It computes the gradient approximation of image intensity function for image edge 
        detection."""
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if filter_name == 'prewittY':
            prewitt_img = cv2.filter2D(img, -1, kernely)
        elif filter_name == 'prewittX':
            prewitt_img = cv2.filter2D(img, -1, kernelx)
        else:
            prewitty_img = cv2.filter2D(img, -1, kernely)
            prewittx_img = cv2.filter2D(img, -1, kernelx)
            prewitt_img = prewitty_img + prewittx_img
        return prewitt_img

    return cv2.filter2D(src=img, ddepth=-1, kernel=kernel)  # (img, cv2.CV_8UC1, kernel)


def gray_image_filters(img, filter_name):
    if filter_name == "gray_scale":                                                     # Gray Scale Image
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif filter_name == "hist_equalize":                                                # Histogram Equalization
        return cv2.equalizeHist(img)

    elif filter_name == "log_trans":                                                    # Log Transformation
        c = 255 / np.log(1 + np.max(img))
        log_image = c * (np.log(img + 1))
        log_image = np.array(log_image, dtype=np.uint8)
        return log_image

    elif filter_name == "bit_plane_slicing":                                            # Bit Plane Slicing
        sliced_images, scale, max_sliced_image, index = [], 1, 0, 0
        for i in range(7):
            sliced_images.append(cv2.bitwise_and(img, scale) * 255)
            scale = scale * 2
            a = sum(sum(sliced_images[i]))
            if a > max_sliced_image:
                max_sliced_image = a
                index = i
        return sliced_images[index]

    elif filter_name == "gray_level_slicing":                                           # Gray Level Slicing
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if 100 < img[i, j] < 180:
                    img[i, j] = 255
        return img

    elif filter_name == "thresh":                                                       # Thresh
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] < 150:
                    img[i, j] = 0
                else:
                    img[i, j] = 255
        return img


def lookup_table(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def rgb_image_filters(img, filter_name):
    if filter_name == "gray2bgr":                                                       # Gray To BGR
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    elif filter_name == "clahe":                            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels
        l2 = clahe.apply(l)  # apply CLAHE to the L-channel

        lab = cv2.merge((l2, a, b))  # merge channels
        clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
        return clahe_img

    elif filter_name == "hdr":                                                          # HDR Effect
        return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)

    elif filter_name == "sepia":                                                        # Sepia Filter
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_sepia = np.array(img, dtype=np.float64)  # Converting to Float to Prevent Loss
        # Multiplying Image With Special Sepia Matrix
        img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168],
                                                        [0.393, 0.769, 0.189]]))
        img_sepia[np.where(img_sepia > 255)] = 255  # Normalizing Values Greater than 255 to 255
        img_sepia = np.array(img_sepia, dtype=np.uint8)
        img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
        return img_sepia

    elif filter_name == "summer":                                                       # Summer Effect
        increase_lookup_table = lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
        decrease_lookup_table = lookup_table([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(img)
        red_channel = cv2.LUT(red_channel, decrease_lookup_table).astype('uint8')
        blue_channel = cv2.LUT(blue_channel, increase_lookup_table).astype('uint8')
        summer = cv2.merge((blue_channel, green_channel, red_channel))
        return summer

    elif filter_name == "winter":                                                       # Winter Effect
        increase_lookup_table = lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
        decrease_lookup_table = lookup_table([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(img)
        red_channel = cv2.LUT(red_channel, increase_lookup_table).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, decrease_lookup_table).astype(np.uint8)
        winter = cv2.merge((blue_channel, green_channel, red_channel))
        return winter

    elif filter_name == "pencil_gray" or filter_name == "pencil_rgb":                   # Pencil Sketch
        sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=20, sigma_r=0.035, shade_factor=0.1)
        if filter_name == "pencil_gray":                                                # Gray Sketch
            return sk_gray
        else:                                                                           # RGB Sketch
            return sk_color

    elif filter_name == "edge_mask":                                                    # Edge Mask
        line_size, blur_value = 7, 7
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, blur_value)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size,
                                      blur_value)
        return edges

    elif filter_name == "color_quantization":                                           # Color Quantization
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        compactness, label, center = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()].reshape(img.shape)
        return result

    elif filter_name == "cartoon":                                                      # Cartoon
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), -1)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
        color = cv2.bilateralFilter(img, d=20, sigmaColor=245, sigmaSpace=245)
        return cv2.bitwise_and(color, color, mask=edges)


def draw_histogram(img):
    if not plt.fignum_exists(1):
        fig = plt.figure(1)
    else:
        plt.close(2)
        fig = plt.figure(2)

    if len(img.shape) == 3:
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
        plt.title('RGB Histogram')
        plt.ylabel('Counts')
        plt.xlabel('Intensity')
        plt.xlim([0, 256])
        plt.show()

    else:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist, color='gray')
        plt.title('Grayscale Histogram')
        plt.ylabel('Counts')
        plt.xlabel('Intensity')
        plt.xlim([0, 256])
        plt.show()


def close_all_figures():
    plt.close('all')


def k_means_edge_detection(img, k=10):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)

    criteria, attempts = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10
    ret, label, center = cv2.kmeans(twoDimage, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape(img.shape)
    return result_image

def edge_detection(img):
    # Edge Detection Process (Compute Gradient approximation and magnitude of vector)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.double(img)  # Convert image to double

    a = np.zeros(shape=img.shape)

    # Prewitt Operator Mask
    Mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    My = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    for i in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 1):
            # Gradient Approximations
            Gy = sum(sum(np.dot(Mx, img[i: i + 3, j: j + 2])))
            Gx = sum(sum(np.dot(My, img[i: i + 3, j: j + 2])))

            # Calculate Magnitude Of Vector
            a[i + 1, j + 1] = math.sqrt(np.power(Gy, 2) + np.power(Gx, 2))

    a = cv2.convertScaleAbs(a, alpha=255 / a.max())
    a = np.uint8(a)
    return a
