from tkinter import Scale, IntVar
from tkinter.ttk import Frame, Button, Label
from cv2 import cv2
import numpy as np
import math


class SegmentToolBox(Frame):
    def __init__(self, master=None):
        super().__init__(master=master)
        self.master = self.master.master.master
        self.image = self.master.image
        self.canny_var, self.kmeans_var, self.thresh_var = IntVar(), IntVar(), IntVar()

        self.Button(self, "Laplace Edge", self.laplace_edge, "", 0, 0, 1)
        self.Button(self, "Outline Edge", self.kernel_filters, "outline_edge", 0, 1, 1)
        self.Button(self, "Sobel Y - 8U", self.sopel, "sobelY", 1, 0, 1)
        self.Button(self, "Sobel Y - 8U(64F)", self.sopel_64f, "sobelY_64f", 1, 1, 1)
        self.Button(self, "Sobel X - 8U", self.sopel, "sobelX", 2, 0, 1)
        self.Button(self, "Sobel X - 8U(64F)", self.sopel_64f, "sobelX_64f", 2, 1, 1)
        self.Button(self, "Sobel Y, X - 8U", self.sopel, "sobelYX", 3, 0, 1)
        self.Button(self, "Sobel Y, X - 8U(64F)", self.sopel_64f, "sobelYX_64f", 3, 1, 1)
        self.Button(self, "Scharr Y", self.scharr, "scharrY", 4, 0, 1)
        self.Button(self, "Scharr X", self.scharr, "scharrX", 4, 1, 1)
        self.Button(self, "Prewitt Y", self.prewitt, "prewittY", 5, 0, 1)
        self.Button(self, "Prewitt X", self.prewitt, "prewittX", 5, 1, 1)
        self.Button(self, "Scharr Y, X", self.scharr, "scharrYX", 6, 0, 1)
        self.Button(self, "Prewitt Y, X", self.prewitt, "prewittYX", 6, 1, 1)
        self.Button(self, "Edge Detection", self.edge_detection, "", 7, 0, 2)
        self.thresh_button = self.Button(self, "Thresholding Mode", self.thresh_mode, "", 14, 0, 2)
        self.Button(self, "AdaptiveT Mean", self.adaptive_thresh, 'thresh_mean', 15, 0, 1)
        self.Button(self, "AdaptiveT Gauss", self.adaptive_thresh, 'thresh_gauss', 15, 1, 1)

        Label(self, text="Thresh Mode").grid(row=13, column=0)
        self.thresh_lbl = Label(self, text="Binary")
        self.thresh_lbl.grid(row=13, column=1)

        self.Scales = [
            self.Scale(self, 0, 255, 4, 'Canny Edge', self.canny_var, self.canny_edge, 10, 0, 2, 0),
            self.Scale(self, 0, 255, 1, 'Thresholding Value', self.thresh_var, self.thresh, 12, 0, 2, 0),
            self.Scale(self, 0, 20, 1, 'K-Means Edges', self.kmeans_var, self.k_means_edge_detection, 16, 0, 2, 0)]

        Label(self, text="").grid(row=20, column=0)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    '''____Widgets_Functions'''
    def disable_elements(self):
        for item in self.winfo_children():
            item.config(state="disabled")

    def enable_scale(self, case_=0):
        if case_ == 0:
            self.Scales[0].update_state("normal")
        elif case_ == 1:
            self.thresh_button.config(state="normal")
            self.Scales[1].update_state("normal")
        elif case_ == 2:
            self.Scales[2].update_state("normal")
        else:
            return

    def enable_elements(self):
        for item in self.winfo_children():
            item.config(state="normal")
        for scale in self.Scales:
            scale.set(0)

    '''____Image_Functions'''
    def refresh_image(self):
        self.image = self.master.image.copy()

    def accept_image(self, ):
        if len(self.image.shape) < 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        self.master.register_image(self.image)
        self.image = self.master.image.copy()

    '''____Buttons_Functions'''
    def kernel_filters(self, mode_):
        kernel = None
        if mode_ == "outline_edge":
            '''Outline Edge Detection'''
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

        self.image = cv2.filter2D(src=self.image, ddepth=-1, kernel=kernel)
        self.accept_image()

    def laplace_edge(self):
        self.image = cv2.Laplacian(self.image, cv2.CV_64F)
        self.image = cv2.convertScaleAbs(self.image)
        self.accept_image()

    def sopel(self, mode_):
        # kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if mode_ == "sobelY":
            self.image = cv2.Sobel(self.image, cv2.CV_8UC1, 0, 1, ksize=3)
        elif mode_ == "sobelX":
            self.image = cv2.Sobel(self.image, cv2.CV_8UC1, 1, 0, ksize=3)
        else:
            sobel_y = cv2.Sobel(self.image, cv2.CV_8U, 0, 1, ksize=3)
            sobel_x = cv2.Sobel(self.image, cv2.CV_8U, 1, 0, ksize=3)
            self.image = sobel_y + sobel_x
        self.image = cv2.convertScaleAbs(self.image)
        self.accept_image()

    def sopel_64f(self, mode_):
        # kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if mode_ == "sobelY_64f":
            self.image = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)
        elif mode_ == "sobelX_64f":
            self.image = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)
        else:
            sobel_y64 = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_x64 = cv2.Sobel(self.image, cv2.CV_8U, 1, 0, ksize=3)
            self.image = sobel_y64 + sobel_x64

        abs_64 = np.absolute(self.image)
        self.image = np.uint8(abs_64)
        self.accept_image()

    def scharr(self, mode_):
        """This Operator Tries to Achieve the Perfect Rotational Symmetry"""
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if mode_ == 'scharrY':
            self.image = cv2.Scharr(self.image, cv2.CV_8UC1, 0, 1)
        elif mode_ == 'scharrX':
            self.image = cv2.Scharr(self.image, cv2.CV_8UC1, 1, 0)
        else:
            scharr_y = cv2.Scharr(self.image, cv2.CV_8UC1, 0, 1)
            scharr_x = cv2.Scharr(self.image, cv2.CV_8UC1, 1, 0)
            self.image = scharr_y + scharr_x
        self.accept_image()

    def prewitt(self, mode_):
        """Prewitt Operator - It is a gradient-based operator. It is one of the best ways to detect the orientation and
        magnitude of an image. It computes the gradient approximation of image intensity function for image edge
        detection"""
        kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if mode_ == 'prewittY':
            self.image = cv2.filter2D(self.image, -1, kernel_y)
        elif mode_ == 'prewittX':
            self.image = cv2.filter2D(self.image, -1, kernel_x)
        else:
            prewitt_y = cv2.filter2D(self.image, -1, kernel_y)
            prewitt_x = cv2.filter2D(self.image, -1, kernel_x)
            self.image = prewitt_y + prewitt_x
        self.accept_image()

    def edge_detection(self):
        """Edge Detection Process (Compute Gradient approximation and magnitude of vector)"""
        self.image = np.uint8(self.image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.image = np.double(self.image)  # Convert image to double

        _img = np.zeros(shape=self.image.shape)

        # Prewitt Operator Mask
        mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        my = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        for i in range(self.image.shape[0] - 2):
            for j in range(self.image.shape[1] - 1):
                # Gradient Approximations
                gy = sum(sum(np.dot(mx, self.image[i: i + 3, j: j + 2])))
                gx = sum(sum(np.dot(my, self.image[i: i + 3, j: j + 2])))

                # Calculate Magnitude Of Vector
                _img[i + 1, j + 1] = math.sqrt(np.power(gy, 2) + np.power(gx, 2))

        self.image = cv2.convertScaleAbs(_img, alpha=255 / np.max(_img))
        self.image = np.uint8(self.image)
        self.accept_image()

    def canny_edge(self, *args):
        if self.master.master.register:
            self.master.master.register = False
            self.master.disable_elements()
            self.enable_scale(0)

        alpha = self.canny_var.get()
        beta = alpha * 2
        self.master.temp_image = cv2.Canny(self.image, alpha, beta)
        self.master.show_temp_image()

    def thresh_mode(self):
        mode_ = self.thresh_lbl.cget("text")
        if mode_ == 'Binary':
            self.thresh_lbl.config(text='Binary_Inv')
        elif mode_ == 'Binary_Inv':
            self.thresh_lbl.config(text='Trunc')
        elif mode_ == 'Trunc':
            self.thresh_lbl.config(text='To_Zero')
        elif mode_ == 'To_Zero':
            self.thresh_lbl.config(text='To_Zero_Inv')
        elif mode_ == 'To_Zero_Inv':
            self.thresh_lbl.config(text='Binary')

        if not self.master.master.register:
            self.thresh()

    def thresh(self, *args):
        if self.master.master.register:
            self.master.master.register = False
            self.master.disable_elements()
            self.enable_scale(1)

        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        value, mode_ = self.thresh_var.get(), self.thresh_lbl.cget("text")
        if mode_ == 'Binary':
            mode = cv2.THRESH_BINARY
        elif mode_ == 'Binary_Inv':
            mode = cv2.THRESH_BINARY_INV
        elif mode_ == 'Trunc':
            mode = cv2.THRESH_TRUNC
        elif mode_ == 'To_Zero':
            mode = cv2.THRESH_TOZERO
        elif mode_ == 'To_Zero_Inv':
            mode = cv2.THRESH_TOZERO_INV

        ret, self.master.temp_image = cv2.threshold(self.image, value, 255, mode+cv2.THRESH_OTSU)
        self.master.show_temp_image()

    def adaptive_thresh(self, case, blocks=7, c=7):
        """
        Adaptive thresholding determines the threshold for a pixel, based on a small region around it. So we get
        different thresholds for different regions of the same image which gives better results for images with
        varying illumination

        cv2.adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)

            - src image that we want to perform thresholding on. This image should be grayscale.
            - dst output array of the same size and type and the same number of channels as src.
            - maxval maximum value which is assigned to pixel values exceeding the threshold
            - adaptiveMethod: adaptiveMethod decides how the threshold value is calculated:
                * cv2.ADAPTIVE_THRESH_MEAN_C: The threshold value is the mean of the neighbourhood area minus the
                    constant C.
                * cv2.ADAPTIVE_THRESH_GAUSSIAN_C: The threshold value is a gaussian-weighted sum of the neighbourhood
                    values minus the constant C.

            - thresholdType: Representing the type of threshold to be used.
            - blockSize: It determines the size of the neighbourhood area.
            - C: C is a constant that is subtracted from the mean or weighted sum of the neighborhood pixels.
        """
        mode_ = self.thresh_lbl.cget("text")
        if mode_ == 'Binary':
            mode = cv2.THRESH_BINARY
        elif mode_ == 'Binary_Inv':
            mode = cv2.THRESH_BINARY_INV
        elif mode_ == 'Trunc':
            mode = cv2.THRESH_TRUNC
        elif mode_ == 'To_Zero':
            mode = cv2.THRESH_TOZERO
        elif mode_ == 'To_Zero_Inv':
            mode = cv2.THRESH_TOZERO_INV

        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if case == "thresh_mean":
            self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, blocks, c)
        else:
            self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, blocks, c)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.accept_image()

    def k_means_edge_detection(self, *args):
        if self.master.master.register:
            self.master.master.register = False
            self.master.disable_elements()
            self.enable_scale(2)

        k = self.kmeans_var.get()
        if k == 0:
            return
        self.master.temp_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        twod_image = self.master.temp_image.reshape((-1, 3))
        twod_image = np.float32(twod_image)

        criteria, attempts = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10
        ret, label, center = cv2.kmeans(twod_image, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        self.master.temp_image = res.reshape(self.master.temp_image.shape)
        self.master.show_temp_image()

    class Button:
        def __init__(self, parent, text, func, param, row, column, cs, state="normal"):
            if param == "":
                self.button = Button(parent, text=text, state=state, command=lambda: func())
            else:
                self.button = Button(parent, text=text, state=state, command=lambda: func(param))
            self.button.grid(row=row, column=column, columnspan=cs, padx=0, pady=0, sticky="nsew")

        def config(self, state):
            self.button.config(state=state)

    class Scale:
        def __init__(self, parent, from_, to, resolution, label, variable, func, row, column,
                     cs=1, init=0, state='normal'):
            self.scale = Scale(parent, orient='horizontal', cursor='hand2', from_=from_, to=to,
                               resolution=resolution, label=label, variable=variable, state=state)
            self.scale.grid(row=row, column=column, columnspan=cs, padx=1, pady=0, sticky="nsew")
            self.scale.set(init)
            variable.trace("w", func)

        def update_state(self, state):
            self.scale.config(state=state)

        def set(self, value):
            self.scale.set(value)
