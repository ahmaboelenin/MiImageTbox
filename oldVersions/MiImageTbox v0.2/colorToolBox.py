from tkinter import Scale, IntVar
from tkinter.ttk import Frame, Button, Label
from cv2 import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline


def lookup_table(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


class ColorToolBox(Frame):
    def __init__(self, master=None):
        super().__init__(master=master)
        self.master = self.master.master.master
        self.image = self.master.image
        self.red_var, self.green_var, self.blue_var = IntVar(), IntVar(), IntVar()

        self.Button(self, "Gray Scale", self.color_to_gray, '', 0, 0, 1)
        self.Button(self, "Histogram Equalize", self.histogram_equalize, '', 0, 1, 1)
        self.Button(self, "Log Transform", self.log_transformation, '', 1, 0, 1)
        self.Button(self, "Bit Plane Slicing", self.bit_plane_slicing, '', 1, 1, 1)
        self.Button(self, "Gray Level Slicing", self.gray_level_slicing, '', 2, 0, 1)
        Label(self, text="").grid(row=5, column=0)

        self.Scales = [
            self.Scale(self, 0, 255, 1, 'Red', self.red_var, self.color_scale, 10, 0, 1, 0),
            self.Scale(self, 0, 255, 1, 'Green', self.green_var, self.color_scale, 10, 1, 1, 0),
            self.Scale(self, 0, 255, 1, 'Blue', self.blue_var, self.color_scale, 11, 0, 1, 0)]
        self.color_button = self.Button(self, "RGB To BGR", self.color_to_color, "", 12, 0, 2)
        self.Button(self, "CLAHE", self.clahe_effect, "", 13, 0, 1)
        self.Button(self, "HDR Effect", self.hdr_effect, "", 13, 1, 1)
        self.Button(self, "Summer Effect", self.summer_effect, "", 14, 0, 1)
        self.Button(self, "Winter Effect", self.winter_effect, "", 14, 1, 1)
        self.Button(self, "Sepia Filter", self.sepia_effect, "", 15, 0, 1)
        self.Button(self, "Cartoon", self.cartoon, "", 15, 1, 1)
        self.Button(self, "Pencil Gary Sketch", self.pencil_sketch, "pencil_gray", 16, 0, 1)
        self.Button(self, "Pencil RGB Sketch", self.pencil_sketch, "pencil_color", 16, 1, 1)
        self.Button(self, "Color Quantization", self.color_quantization, "", 17, 0, 1)
        Label(self, text="").grid(row=20, column=0)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    '''____Widgets_Functions'''

    def disable_elements(self):
        for item in self.winfo_children():
            item.config(state="disabled")

    def enable_scale(self):
        for scale in self.Scales:
            scale.update_state("normal")

    def enable_elements(self):
        for item in self.winfo_children():
            item.config(state="normal")
        for scale in self.Scales:
            scale.set(0)

    '''____Image_Functions'''
    def refresh_image(self):
        self.image = self.master.image.copy()

    def accept_image(self):
        self.master.register_image(self.image)
        self.image = self.master.image.copy()

    '''____Buttons_Functions'''
    def color_to_gray(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.accept_image()

    def histogram_equalize(self):
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.image = cv2.equalizeHist(self.image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.accept_image()

    def log_transformation(self):
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        c = 255 / np.log(1 + np.max(self.image))
        self.image = c * (np.log(self.image + 1))
        self.image = np.array(self.image, dtype=np.uint8)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.accept_image()

    def bit_plane_slicing(self, n=None):
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if n is None:  # Get Maximum Bit
            sliced_images, scale, max_sliced_image, index = [], 1, 0, 0
            for i in range(7):
                sliced_images.append(cv2.bitwise_and(self.image, scale) * 255)
                scale = scale * 2
                a = sum(sum(sliced_images[i]))
                if a > max_sliced_image:
                    max_sliced_image = a
                    index = i
            self.image = sliced_images[index]
        else:
            self.image = cv2.bitwise_and(self.image, pow(2, n)) * 255
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.accept_image()

    def gray_level_slicing(self, a=100, b=180):
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if a < self.image[i, j] < b:
                    self.image[i, j] = 255
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        self.accept_image()

    def color_scale(self, *args):
        if self.master.master.register:
            self.master.master.register = False
            self.master.disable_elements()
            self.enable_scale()

        alpha_r, alpha_g, alpha_b = self.red_var.get(), self.green_var.get(), self.blue_var.get()
        self.master.temp_image = cv2.addWeighted(self.image, 1, self.image, 0, 0)
        if alpha_r != 0:
            red_img = np.full((self.image.shape[0], self.image.shape[1], 3), (alpha_r, 0, 0), np.uint8)
            self.master.temp_image = cv2.addWeighted(self.master.temp_image, 0.8, red_img, 0.2, 0)
        if alpha_g != 0:
            green_img = np.full((self.image.shape[0], self.image.shape[1], 3), (0, alpha_g, 0), np.uint8)
            self.master.temp_image = cv2.addWeighted(self.master.temp_image, 0.8, green_img, 0.2, 0)
        if alpha_b != 0:
            blue_img = np.full((self.image.shape[0], self.image.shape[1], 3), (0, 0, alpha_b), np.uint8)
            self.master.temp_image = cv2.addWeighted(self.master.temp_image, 0.8, blue_img, 0.2, 0)
        self.master.show_temp_image()

    def color_to_color(self):
        mode_ = self.color_button.get_text()
        if mode_ == 'RGB To BGR':
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            self.color_button.set_text('BGR To RGB')
        else:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.color_button.set_text('RGB To BGR')
        self.accept_image()

    def clahe_effect(self):
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels
        l2 = clahe.apply(l)  # apply CLAHE to the L-channel

        lab = cv2.merge((l2, a, b))  # merge channels
        self.image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
        self.accept_image()

    def hdr_effect(self):
        self.image = cv2.detailEnhance(self.image, sigma_s=12, sigma_r=0.15)
        self.accept_image()

    def summer_effect(self):
        increase_lookup_table = lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
        decrease_lookup_table = lookup_table([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(self.image)
        red_channel = cv2.LUT(red_channel, decrease_lookup_table).astype('uint8')
        blue_channel = cv2.LUT(blue_channel, increase_lookup_table).astype('uint8')
        self.image = cv2.merge((blue_channel, green_channel, red_channel))
        self.accept_image()

    def winter_effect(self):
        increase_lookup_table = lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
        decrease_lookup_table = lookup_table([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(self.image)
        red_channel = cv2.LUT(red_channel, increase_lookup_table).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, decrease_lookup_table).astype(np.uint8)
        self.image = cv2.merge((blue_channel, green_channel, red_channel))
        self.accept_image()

    def sepia_effect(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        self.image = np.array(self.image, dtype=np.float64)  # Converting to Float to Prevent Loss

        # Multiplying Image With Special Sepia Matrix
        self.image = cv2.transform(self.image, np.matrix([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168],
                                                          [0.393, 0.769, 0.189]]))
        self.image[np.where(self.image > 255)] = 255  # Normalizing Values Greater than 255 to 255
        self.image = np.array(self.image, dtype=np.uint8)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.accept_image()

    def cartoon(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), -1)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
        color = cv2.bilateralFilter(self.image, d=20, sigmaColor=245, sigmaSpace=245)
        self.image = cv2.bitwise_and(color, color, mask=edges)
        self.accept_image()

    def pencil_sketch(self, mode):
        sk_gray, sk_color = cv2.pencilSketch(self.image, sigma_s=20, sigma_r=0.035, shade_factor=0.1)
        if mode == "pencil_gray":  # Pencil Sketch
            self.image = sk_gray
        else:  # Color Sketch
            self.image = sk_color
        self.accept_image()

    def color_quantization(self):
        data = np.float32(self.image).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        compactness, label, center = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        self.image = center[label.flatten()].reshape(self.image.shape)
        self.accept_image()

    class Button:
        def __init__(self, parent, text, func, param, row, column, cs, state="normal"):
            if param == "":
                self.button = Button(parent, text=text, state=state, command=lambda: func())
            else:
                self.button = Button(parent, text=text, state=state, command=lambda: func(param))
            self.button.grid(row=row, column=column, columnspan=cs, padx=0, pady=0, sticky="nsew")

        def get_text(self):
            return self.button.cget('text')

        def set_text(self, text):
            self.button.config(text=text)

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
