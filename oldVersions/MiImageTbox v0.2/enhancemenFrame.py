from tkinter import Scale, IntVar, DoubleVar
from tkinter.ttk import Frame, Button, Label
import cv2.cv2 as cv2
import numpy as np
import random
import math


class EnhanceToolBox(Frame):
    def __init__(self, master=None):
        super().__init__(master=master)
        self.master = self.master.master.master
        self.image = self.master.image
        self.brightness_var, self.contrast_var, self.gamma_var = IntVar(), IntVar(), DoubleVar()

        self.Scales = [
            self.Scale(self, 0, 510, 1, 'Brightness', self.brightness_var, self.brightness_contrast, 0, 0, 2,
                       255),
            self.Scale(self, 0, 254, 1, 'Contrast', self.contrast_var, self.brightness_contrast, 1, 0, 2, 127),
            self.Scale(self, 0.1, 2, 0.1, 'Gamma', self.gamma_var, self.gamma, 2, 0, 2, 1)]

        self.Button(self, "Adjust Gamma", self.gamma, None, 3, 0, 1)
        self.Button(self, "Salt & Pepper", self.salt_pepper, "", 3, 1, 1)
        self.Button(self, "Negative", self.negative, "", 4, 0, 1)
        self.Button(self, "Bilateral", self.kernel_filters, "bilateral", 4, 1, 1)
        self.Button(self, "Blurring", self.kernel_filters, "blur", 5, 0, 1)
        self.Button(self, "Median Blurring", self.kernel_filters, "median", 5, 1, 1)
        self.Button(self, "Gaussian Filter", self.kernel_filters, "gaussian", 6, 0, 1)
        self.Button(self, "Pyramidal Filter", self.kernel_filters, "pyramidal", 6, 1, 1)
        self.Button(self, "Circular Filter", self.kernel_filters, "circular", 7, 0, 1)
        self.Button(self, "Cone Filter", self.kernel_filters, "cone", 7, 1, 1)
        self.Button(self, "Emboss Filter", self.kernel_filters, "emboss", 8, 0, 1)
        self.Button(self, "Sharpen Filter", self.kernel_filters, "sharpen", 8, 1, 1)
        self.Button(self, "Add Border", self.add_border, "", 9, 0, 1)
        Label(self, text="").grid(row=20, column=0)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    '''____Widgets_Functions'''
    def disable_elements(self):
        for item in self.winfo_children():
            item.config(state="disabled")

    def enable_scale(self, case_):
        if case_ == 0:
            self.Scales[0].update_state("normal")
            self.Scales[1].update_state("normal")
        else:
            self.Scales[2].update_state("normal")

    def enable_elements(self):
        for item in self.winfo_children():
            item.config(state="normal")

        self.Scales[0].set(255)
        self.Scales[1].set(127)
        self.Scales[2].set(1)

    '''____Image_Functions'''
    def refresh_image(self):
        self.image = self.master.image.copy()

    def accept_image(self):
        if len(self.image.shape) < 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        self.master.register_image(self.image)
        self.image = self.master.image.copy()

    '''____Buttons_Functions'''
    def brightness_contrast(self, *args):
        if self.master.master.register:
            self.master.master.register = False
            self.master.disable_elements()
            self.enable_scale(0)

        brightness, contrast = self.brightness_var.get(), self.contrast_var.get()
        brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
        contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha = (highlight - shadow) / 255
            gamma = shadow
            self.master.temp_image = cv2.addWeighted(self.master.image, alpha, self.master.image, 0, gamma)
        else:
            self.master.temp_image = self.master.image

        if contrast != 0:
            alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            gamma = 127 * (1 - alpha)
            self.master.temp_image = cv2.addWeighted(self.master.temp_image, alpha, self.master.temp_image, 0, gamma)
        if brightness == 0 and contrast == 0:
            return
        self.master.show_temp_image()

    def gamma(self, alpha=0, *args):
        image = self.image
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if alpha is None:
            mid, mean = 0.5, np.mean(gray)
            gamma = math.log(mid * 255) / math.log(mean)
            self.image = np.power(image, gamma).clip(0, 255).astype(np.uint8)
            self.accept_image()

        else:
            if self.master.master.register:
                self.master.master.register = False
                self.master.disable_elements()
                self.enable_scale(1)

            gamma = self.gamma_var.get()
            self.master.temp_image = np.power(image, gamma).clip(0, 255).astype(np.uint8)
            self.master.show_temp_image()

    def salt_pepper(self):
        row, col = self.image.shape[:2]

        number_of_pixels = random.randint(300, 10000)  # Pick a random number between 300 and 10000
        for i in range(number_of_pixels):
            y_coord = random.randint(0, row - 1)  # Pick a random y coordinate
            x_coord = random.randint(0, col - 1)  # Pick a random x coordinate
            self.image[y_coord][x_coord] = 255  # Color that pixel to white

        number_of_pixels = random.randint(300, 10000)  # Pick a random number between 300 and 10000
        for i in range(number_of_pixels):
            y_coord = random.randint(0, row - 1)  # Pick a random y coordinate
            x_coord = random.randint(0, col - 1)  # Pick a random x coordinate
            self.image[y_coord][x_coord] = 0  # Color that pixel to black
        self.accept_image()

    def negative(self):
        self.image = cv2.bitwise_not(self.master.image)
        self.accept_image()

    def kernel_filters(self, mode_):
        kernel = None
        if mode_ == "blur":
            '''(Traditional / Average) Filter'''
            kernel = np.ones([3, 3]) / 9
            # self.image = cv2.blur(src=img, ksize=(3, 3))

        elif mode_ == "median":
            '''Median Blurring'''
            self.image = cv2.medianBlur(src=self.master.image, ksize=3)
            self.accept_image()
            return

        elif mode_ == "gaussian":
            '''Gaussian Filter'''
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
            # self.image = cv2.GaussianBlur(img, (5, 5), sigmaX=0)

        elif mode_ == "pyramidal":
            '''Pyramidal Filter'''
            kernel = np.array(
                [[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3], [2, 4, 6, 4, 2], [1, 2, 3, 2, 1]]) / 81

        elif mode_ == "circular":
            '''Circular Filter'''
            kernel = np.array(
                [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]]) / 21

        elif mode_ == "cone":
            '''Cone Filter'''
            kernel = np.array(
                [[0, 0, 1, 0, 0], [0, 2, 2, 2, 0], [1, 2, 5, 2, 1], [0, 2, 2, 2, 0], [0, 0, 1, 0, 0]]) / 25

        elif mode_ == "bilateral":
            '''Bilateral Filter'''
            self.image = cv2.bilateralFilter(src=self.master.image, d=9, sigmaColor=75, sigmaSpace=75)
            self.accept_image()
            return

        elif mode_ == "emboss":
            '''Emboss Filter'''
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

        elif mode_ == "sharpen":
            '''Sharpening Filter'''
            # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])

        self.image = cv2.filter2D(src=self.master.image, ddepth=-1, kernel=kernel)  # ddepth=cv2.CV_8UC1, kernel
        self.accept_image()

    def add_border(self, color=[0, 0, 0]):
        # temp = cv2.copyMakeBorder(self.master.image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=color)
        temp = cv2.copyMakeBorder(self.master.image, 10, 10, 10, 10, cv2.BORDER_REFLECT)
        self.image = cv2.resize(temp, (self.master.image.shape[1], self.master.image.shape[0]))
        self.accept_image()

    class Button:
        def __init__(self, parent, text, func, param, row, column, cs, state="normal"):
            if param == "":
                self.button = Button(parent, text=text, state=state, command=lambda: func())
            else:
                self.button = Button(parent, text=text, state=state, command=lambda: func(param))
            self.button.grid(row=row, column=column, columnspan=cs, padx=0, pady=0, sticky="nsew")

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
