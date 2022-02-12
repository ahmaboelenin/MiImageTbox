from tkinter import Canvas, Scale, filedialog, IntVar, DoubleVar
from tkinter.ttk import Frame, Button, Label, Entry, Separator
import cv2.cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

from enhancemenFrame import EnhanceToolBox
from colorToolBox import ColorToolBox
from segmentToolBox import SegmentToolBox


def draw_histogram(image):
    if not plt.fignum_exists(1):
        fig = plt.figure(1)
    else:
        plt.close(2)
        fig = plt.figure(2)

    if len(image.shape) == 3:
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title('RGB Histogram')
        plt.ylabel('Counts')
        plt.xlabel('Intensity')
        plt.xlim([0, 256])
        plt.show()

    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='gray')
        plt.title('Grayscale Histogram')
        plt.ylabel('Counts')
        plt.xlabel('Intensity')
        plt.xlim([0, 256])
        plt.show()


def close_all_figures():
    plt.close('all')


def is_number(num):
    """This Function Check if the Entry is Number or Not to Avoid Crashing."""
    try:
        return int(num)
    except ValueError:
        return False


def browse_files():
    """This Function Open File Dialog to Read the New Image."""
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File",
                                          filetypes=(("Image files", "*.jpg *.jpeg *png"), ("all files", "*.*")))
    return filename


class ToolBox(Frame):
    def __init__(self, master=None):
        self.max_width, self.max_height = int(master.screen_width * 0.13), int(master.screen_height * 0.95)
        self.cen = int(self.max_width / 2)
        super().__init__(master=master, width=self.max_width, height=self.max_height)

        self.path, self.image, self.temp_image, self.blendImage, self.maskImage = \
            None, self.master.image, None, None, None
        self.translateX_var, self.translateY_var, self.blind_var = IntVar(), IntVar(), DoubleVar()

        self.mainTool = Frame(self, borderwidth=0)

        self.actual_size_label = self.Label(self.mainTool, "Actual Size: ", 0, 0, 2)
        self.view_size_label = self.Label(self.mainTool, "View Size: ", 1, 0, 2)
        Separator(self.mainTool, orient='horizontal').grid(row=2, column=0, columnspan=2, pady=5, sticky='nsew')

        self.new_button = self.Button(self.mainTool, "New Image", self.open, "", 3, 0, 2, img="img/open.png")
        self.Button(self.mainTool, "Save", self.save, "", 4, 0, 1)
        self.Button(self.mainTool, "Save As", self.save_as, "", 4, 1, 1)
        self.undo_button = self.Button(self.mainTool, "Undo", self.undo, "", 5, 0, 1, "disabled")
        self.redo_button = self.Button(self.mainTool, "Redo", self.redo, "", 5, 1, 1, "disabled")
        self.done_button = self.Button(self.mainTool, "Done", self.done, "", 6, 0, 1, "disabled")
        self.cancel_button = self.Button(self.mainTool, "Cancel", self.cancel, "", 6, 1, 1, "disabled")
        self.original_button = self.Button(self.mainTool, "Show Original", self.show_original, "", 7, 0, 1, "disabled")
        self.Button(self.mainTool, "Get Histogram", lambda: draw_histogram(self.master.image), "", 7, 1, 1)
        self.Button(self.mainTool, "Save As Mask", self.load_mask_image, "", 8, 0, 1)
        self.Button(self.mainTool, "Use Mask", self.use_mask_image, "", 8, 1, 1)
        self.image_state = self.Label(self.mainTool, "No Image Loaded", 9, 0, 2)
        self.Label(self.mainTool, "", 10, 0, 2)
        Separator(self.mainTool, orient='horizontal').grid(row=11, column=0, columnspan=2, pady=2, sticky='nsew')

        '''____Tool_Box_Canvas'''
        self.tool_box_canvas = Canvas(self, borderwidth=0, highlightthickness=0, width=self.max_width)
        self.tool_box = Frame(self.tool_box_canvas, borderwidth=0)

        self.entry_width = self.Entry(self.tool_box, "Width", lambda x: self.get_height(None), self.reset_width, 0, 0)
        self.entry_height = self.Entry(self.tool_box, "Height", lambda x: self.get_width(None), self.reset_height, 0, 1)
        self.Button(self.tool_box, "Resize", self.resize, "", 1, 0, 2)
        Label(self.tool_box, text="").grid(row=2, column=0)

        self.Button(self.tool_box, "Rotate Left", self.rotate, 90, 3, 0, 1)
        self.Button(self.tool_box, "Rotate Right", self.rotate, -90, 3, 1, 1)
        self.Button(self.tool_box, "Vertical Flip", self.flip, 1, 4, 0, 1)
        self.Button(self.tool_box, "Horizontal Flip", self.flip, 0, 4, 1, 1)
        self.Button(self.tool_box, "Rotate 180° Or Flip XY", self.rotate, 180, 5, 0, 2)
        self.Button(self.tool_box, "Crop", self.crop, "", 6, 0, 1)
        self.Button(self.tool_box, "4 Points Transform", self.four_points_transform, "", 6, 1, 1)
        self.Button(self.tool_box, "Zoom", self.zoom, "", 7, 0, 1)
        self.Button(self.tool_box, "Skew", self.skew, "", 7, 1, 1)

        self.Button(self.tool_box, "Un-Translate", self.un_translate, "", 9, 0, 1)
        self.Button(self.tool_box, "Un-Skewing", self.un_skewing, "", 9, 1, 1)
        self.Button(self.tool_box, "Skewing", self.skewing, "", 10, 0, 1)

        self.Button(self.tool_box, "Load Blending Image", self.load_blend, "", 11, 0, 2)

        self.Scales = [
            self.Scale(self.tool_box, -100, 100, 10, 'Translate-X', self.translateX_var, self.translate, 8, 0, 1, 0),
            self.Scale(self.tool_box, -100, 100, 10, 'Translate-Y', self.translateY_var, self.translate, 8, 1, 1, 0),
            self.Scale(self.tool_box, 0, 1, 0.1, 'Blending', self.blind_var, self.blend, 12, 0, 2, 0, 'disabled')]
        Label(self.tool_box, text="").grid(row=13, column=0)
        Separator(self.tool_box, orient='horizontal').grid(row=14, column=0, columnspan=2, pady=5, sticky='nsew')

        self.enhancement_section = EnhanceToolBox(self.tool_box)
        self.enhancement_section.grid(row=20, column=0, columnspan=2, sticky='nsew')
        Separator(self.tool_box, orient='horizontal').grid(row=22, column=0, columnspan=2, pady=5, sticky='nsew')

        self.colors_section = ColorToolBox(self.tool_box)
        self.colors_section.grid(row=23, column=0, columnspan=2, sticky='nsew')
        Separator(self.tool_box, orient='horizontal').grid(row=25, column=0, columnspan=2, pady=5, sticky='nsew')

        self.segmentation_button = SegmentToolBox(self.tool_box)
        self.segmentation_button.grid(row=26, column=0, columnspan=2, sticky='nsew')
        Separator(self.tool_box, orient='horizontal').grid(row=28, column=0, columnspan=2, pady=2, sticky='nsew')

        self.tool_box_canvas.create_window((0, 0), window=self.tool_box, anchor="nw")
        self.tool_box_canvas.bind('<Enter>', self._bound_to_mousewheel)
        self.tool_box_canvas.bind('<Leave>', self._unbound_to_mousewheel)

        self.tool_box.bind("<Configure>",
                           lambda x: self.tool_box_canvas.configure(scrollregion=self.tool_box_canvas.bbox("all")))

        self.mainTool.grid_columnconfigure(0, weight=1)
        self.mainTool.grid_columnconfigure(1, weight=1)

        self.mainTool.pack(fill='both')
        self.tool_box_canvas.pack(fill='both', expand=1)

    '''____Binding_Functions'''

    def _on_mousewheel(self, event):  # ScrollBar Movement with Mouse Wheel
        self.tool_box_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bound_to_mousewheel(self, event):
        self.tool_box_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        self.tool_box_canvas.unbind_all("<MouseWheel>")

    def init(self):
        self.disable_all_elements()
        self.new_button.config('normal')

    '''____Widgets_Functions'''

    def disable_all_elements(self):
        for item in self.winfo_children():
            item.config(state="disabled")

    def disable_elements(self):
        for item in self.mainTool.children.values():
            if item.winfo_class() == 'TButton':
                item.config(state="disabled")

        for item in self.tool_box.children.values():
            if item.winfo_class() in ['TButton', 'Scale']:
                item.config(state="disabled")

        self.enhancement_section.disable_elements()
        self.colors_section.disable_elements()
        self.segmentation_button.disable_elements()

        self.done_button.config("normal")
        self.cancel_button.config("normal")

    def enable_scale(self, case_):
        if case_ == 0:
            self.Scales[0].update_state("normal")
            self.Scales[1].update_state("normal")
        elif case_ == 1:
            self.Scales[2].update_state("normal")
        elif case_ == 2:
            self.original_button.config("normal")
            self.done_button.config("disabled")
            self.cancel_button.config("disabled")
        else:
            return

    def enable_elements(self):
        for item in self.mainTool.children.values():
            if item.winfo_class() == 'TButton':
                item.config(state="normal")

        for item in self.tool_box.children.values():
            if item.winfo_class() in ['TButton', 'Scale']:
                item.config(state="normal")

        self.enhancement_section.enable_elements()
        self.colors_section.enable_elements()
        self.segmentation_button.enable_elements()

        self.done_button.config("disabled")
        self.cancel_button.config("disabled")
        if len(self.master.imageHistory) == 1:
            self.undo_button.config("disabled")
        if self.master.index >= len(self.master.imageHistory) - 1:
            self.redo_button.config("disabled")

        self.Scales[0].set(0)
        self.Scales[1].set(0)
        self.Scales[2].set(0)
        self.Scales[2].update_state("disabled")
        try:
            self.tool_box.enable_elements()
        except AttributeError:
            return

    '''____Image_Functions'''

    def show_temp_image(self):
        self.master.imageBox.show_image(self.temp_image)

    def register_image(self, image):
        if self.register:
            if self.master.undo:
                self.master.undo = False
                self.redo_button.config("disabled")
                for i in range(self.master.index + 1, len(self.master.imageHistory)):
                    self.master.imageHistory.pop(-1)
            self.master.index += 1
            if self.master.index > 0:
                self.original_button.config("normal")
                self.undo_button.config("normal")
            else:
                self.original_button.config("disabled")
                self.undo_button.config("disabled")
            self.master.imageHistory.append(image)
            self.master.image = image
            self.image = self.master.image
            self.refresh_image()
            self.master.imageBox.show_image()

    def refresh_image(self):
        self.enhancement_section.refresh_image()
        self.colors_section.refresh_image()
        self.segmentation_button.refresh_image()

    '''____Buttons_Required_Functions'''

    def reset_width(self):
        if self.entry_width.get() == 'Width':
            self.entry_width.reset()
            self.entry_width.unbind()

    def reset_height(self):
        if self.entry_height.get() == 'Height':
            self.entry_height.reset()
            self.entry_height.unbind()

    def reset_entries(self):
        self.entry_width.reset()
        self.entry_width.set('Width')
        self.entry_height.reset()
        self.entry_height.set('Height')

    def get_height(self, height):
        if height is None:
            width = is_number(self.entry_width.get())
            height = int(width / self.master.ratio)
        self.entry_height.reset()
        self.entry_height.set(height)
        return height

    def get_width(self, width):
        if width is None:
            height = is_number(self.entry_height.get())
            width = int(height * self.master.ratio)
        self.entry_width.reset()
        self.entry_width.set(width)
        return width

    '''____Buttons_Functions'''

    def open(self, case_=None):
        if case_ == 'Sample':
            # Load Sample Image
            path = 'img/sample.jpg'
        else:
            path = browse_files()
        if path is None:
            return
        self.path = path
        image = cv2.imread(path)
        if image is not None:
            close_all_figures()
            self.master.imageHistory, self.master.index = [], -1
            self.master.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image = self.master.image
            self.refresh_image()
            self.clear_frame()
            self.register_image(self.master.image)
            self.enable_elements()

    def save(self):
        try:
            image = cv2.cvtColor(self.master.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.path, image)
        except:
            ""

    def save_as(self):
        path = filedialog.asksaveasfilename(initialfile='Untitled.jpg', defaultextension=".jpg",
                                            filetypes=[("jpg Image", "*.jpg"), ("jpeg Image", "*.jpeg"),
                                                       ("All Files", "*.*")])
        try:
            image = cv2.cvtColor(self.master.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, image)
        except:
            ""

    def undo(self):
        self.master.undo, self.master.index = True, self.master.index - 1
        self.master.image = self.master.imageHistory[self.master.index]
        self.image = self.master.image
        self.refresh_image()
        if self.master.index == 0:
            self.undo_button.config("disabled")
        self.redo_button.config("normal")
        self.master.imageBox.show_image()

    def redo(self):
        self.master.index = self.master.index + 1
        self.master.image = self.master.imageHistory[self.master.index]
        self.image = self.master.image
        self.refresh_image()
        if self.master.index == len(self.master.imageHistory) - 1:
            self.redo_button.config("disabled")
        self.undo_button.config("normal")
        self.master.imageBox.show_image()

    def reset(self, event=None):
        if self.master.crop:
            self.master.imageBox.reset()
        self.enable_elements()

    def done(self):
        temp = self.temp_image
        self.reset()
        self.master.register = True
        self.register_image(temp)

    def cancel(self):
        self.reset()
        self.master.register = True
        self.master.imageBox.show_image()
        if self.master.crop:
            self.master.imageBox.reset()

    def show_original(self):
        if self.original_button.get_text() == "Show Original":
            self.original_button.set_text("Show Current")
            self.disable_elements()
            self.enable_scale(2)
            self.master.imageBox.show_image(self.master.imageHistory[0])

        else:
            self.original_button.set_text("Show Original")
            self.master.imageBox.show_image()
            self.reset()

    def load_mask_image(self):
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.maskImage = self.image.copy()
        self.image_state.config("Image Loaded")

    def use_mask_image(self):
        self.image = cv2.bitwise_or(self.image, self.image, mask=self.maskImage)

        # self.image = cv2.addWeighted(self.image, 0.8, self.maskImage, 0, 0.2)
        self.register_image(self.image)

    def resize(self):
        width, height = is_number(self.entry_width.get()), is_number(self.entry_height.get())
        if (width is False and height is False) or (width == 0 or height == 0):
            return
        if width is False:
            width = self.get_width(None)
        if height is False or width / height != self.master.ratio:
            height = self.get_height(None)
        self.image = cv2.resize(self.image, (width, height))
        self.reset_entries()
        self.register_image(self.image)

    def force_resize(self, width, height):
        self.image = cv2.resize(self.image, (width, height))
        self.register_image(self.image)

    def rotate(self, angle):  # Rotate Button Function
        height, width = self.image.shape[:2]
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
        self.image = cv2.warpAffine(self.image, rotation_mat, (bound_w, bound_h))
        self.register_image(self.image)

    def flip(self, case_):  # Flip Button Function
        self.image = cv2.flip(self.image, case_)
        self.register_image(self.image)

    def translate(self, *args):
        if self.master.register:
            self.master.register = False
            self.disable_elements()
            self.enable_scale(0)

        shift_x, shift_y = self.translateX_var.get(), self.translateY_var.get()
        mat = np.float32([[1, 0, -shift_x], [0, 1, shift_y]])
        self.temp_image = cv2.warpAffine(self.master.image, mat,
                                         (self.master.image.shape[1], self.master.image.shape[0]))
        # The function warpAffine transforms the source image using the specified matrix
        self.show_temp_image()

    def un_translate(self):
        image, x_start, y_start, x_end, y_end = self.master.image, -1, -1, 0, 0
        for y in range(image.shape[0]):
            if y_start == -1:
                if not np.all(image[y, 0:] == 0):
                    y_start = y
            else:
                if np.all(image[y, 0:] == 0):
                    y_end = y
                    break
        if y_end == 0:
            y_end = image.shape[0]

        for x in range(image.shape[1]):
            if x_start == -1:
                if not np.all(image[0:, x] == 0):
                    x_start = x
                continue
            else:
                if np.all(image[0:, x] == 0):
                    x_end = x
                    break
        if x_end == 0:
            x_end = image.shape[1]

        self.image = image[y_start: y_end, x_start: x_end]
        if np.all(self.image == 0):
            return
        self.image = cv2.resize(self.image, (image.shape[1], image.shape[0]))
        self.register_image(self.image)

    def skewing(self):
        pts1 = np.float32([[0, 0], [self.image.shape[1] - 1, 0], [100, self.image.shape[0] - 1]])
        pts2 = np.float32([[0, 0], [self.image.shape[1] - 1, 0], [0, self.image.shape[0] - 1]])
        sm = cv2.getAffineTransform(pts1, pts2)
        self.image = cv2.warpAffine(self.image, sm, self.image.shape[1::-1])
        self.register_image(self.image)

    def un_skewing(self):
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh_ = cv2.threshold(self.image, 127, 255, 0)[1]
        contours, hierarchy = cv2.findContours(thresh_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # Get Contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        imx = self.image.shape[0]
        imy = self.image.shape[1]
        lp_area = (imx * imy) / 10

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.contourArea(cnt) > lp_area:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                for i in range(len(box)):
                    if box[i, 0] > self.image.shape[1] or box[i, 0] < 0:
                        if box[i, 0] > self.image.shape[1]:
                            box[i, 0] = self.image.shape[1]
                        if box[i, 0] < 0:
                            box[i, 0] = 0
                for i in range(len(box)):
                    if box[i, 1] > self.image.shape[0] or box[i, 1] < 0:
                        if box[i, 1] > self.image.shape[0]:
                            box[i, 1] = self.image.shape[0]
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
                self.image = cv2.warpPerspective(self.image, mat, (max_width, max_height))
        self.register_image(self.image)

    def load_blend(self):
        path = browse_files()
        if path is None:
            return
        image = cv2.imread(path)
        if image is not None:
            self.blendImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.master.register:
            self.master.register = False
            self.disable_elements()
            self.enable_scale(1)

    def blend(self, *args):
        alpha = self.blind_var.get()
        beta = 1 - alpha
        self.blendImage = cv2.resize(self.blendImage, (self.image.shape[1], self.image.shape[0]))
        self.temp_image = cv2.addWeighted(self.image, beta, self.blendImage, alpha, 0.0)
        self.show_temp_image()

    def crop(self):
        self.disable_elements()
        self.done_button.config("disabled")
        self.master.imageBox.start_crop()

    def end_crop(self, x_start, y_start, x_end, y_end):
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            if x_start < 0:
                x_start = 0
            if x_end > self.image.shape[1]:
                x_end = self.image.shape[1]
        if y_start > y_end:
            y_start, y_end = y_end, y_start
            if y_start < 0:
                y_start = 0
            if y_end > self.image.shape[0]:
                y_end = self.image.shape[0]

        image = self.image[y_start: y_end, x_start: x_end]
        self.image = cv2.resize(image, (self.image.shape[1], self.image.shape[0]))
        self.register_image(self.image)
        self.enable_elements()

    def skew(self):
        self.disable_elements()
        self.done_button.config("disabled")
        self.master.imageBox.start_skew()

    def end_skew(self, x, y):
        x = self.image.shape[1] - x
        pts1 = np.float32([[0, 0], [self.image.shape[1] - 1, 0], [x, self.image.shape[0] - 1]])
        pts2 = np.float32([[0, 0], [self.image.shape[1] - 1, 0], [0, self.image.shape[0] - 1]])
        sm = cv2.getAffineTransform(pts1, pts2)
        self.image = cv2.warpAffine(self.image, sm, self.image.shape[1::-1])
        self.register_image(self.image)
        self.enable_elements()
        print(x)

    def four_points_transform(self):
        self.disable_elements()
        self.done_button.config("disabled")
        self.master.imageBox.start_four_points_transform()

    def end_four_points_transform(self, rect):
        (tl, tr, br, bl) = rect
        max_width = max(int(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))),
                        int(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))))
        max_height = max(int(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))),
                         int(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))))

        dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
                       dtype="float32")

        data = np.array(rect, dtype=np.float32)
        mat = cv2.getPerspectiveTransform(data, dst)
        image = cv2.warpPerspective(self.image, mat, (max_width, max_height))
        self.image = cv2.resize(image, (self.image.shape[1], self.image.shape[0]))
        self.register_image(self.image)
        self.enable_elements()

    def zoom(self):
        self.disable_elements()
        self.done_button.config("disabled")
        self.master.imageBox.start_zoom()

    def end_zoom(self, x, y):
        y_max, x_max = int(self.image.shape[0] / 4), int(self.image.shape[1] / 4)
        y_start, y_end, x_start, x_end = y - y_max, y + y_max, x - x_max, x + x_max

        if y_start < 0:
            y_start = 0
        if y_end > self.image.shape[0]:
            y_end = self.image.shape[0]

        if x_start < 0:
            x_start = 0
        if x_end > self.image.shape[1]:
            x_end = self.image.shape[1]

        image = self.image[y_start: y_end, x_start: x_end]
        if np.all(image == 0):
            return
        self.image = cv2.resize(image, (self.image.shape[1], self.image.shape[0]))
        self.register_image(self.image)
        self.enable_elements()

    '''____Frame_Functions'''

    def clear_frame(self):
        # self.tool_box_canvas.delete("all")
        ""

    class Button:
        def __init__(self, parent, text, func, param, row, column, cs, state="normal", img=""):
            # ima = PhotoImage(file=img).subsample(12)
            if param == "":
                self.button = Button(parent, text=text, state=state, command=lambda: func())
            else:
                self.button = Button(parent, text=text, state=state, command=lambda: func(param))
            # image=ima, compound=tk.LEFT,
            # self.button.image = ima
            self.button.grid(row=row, column=column, columnspan=cs, padx=0, pady=0, sticky="nsew")

        def config(self, state):
            self.button.config(state=state)

        def get_text(self):
            return self.button.cget('text')

        def set_text(self, text):
            self.button.config(text=text)

    class Entry:
        def __init__(self, parent, text, func, func2, row, column):
            self.entry = Entry(parent)
            self.entry.insert(0, text)
            self.entry.bind('<Return>', func)
            self.entry.bind('<Button-1>', lambda x: func2())
            self.entry.grid(row=row, column=column, padx=1, pady=1, sticky="nsew")

        def get(self):
            return self.entry.get()

        def reset(self):
            self.entry.delete(0, 'end')

        def set(self, text):
            self.entry.insert(0, text)

        def unbind(self):
            self.entry.unbind('<Button-1>')

    class Label:
        def __init__(self, parent, text, row, column, cs):
            self.label = Label(parent, anchor='center')
            self.label.config(text=text)
            self.label.grid(row=row, column=column, columnspan=cs, sticky="w", padx=2, pady=1)

        def config(self, txt):
            self.label.config(text=txt)

    class Scale:
        def __init__(self, parent, from_, to, resolution, label, variable, func, row, column,
                     cs=1, init=0, state='normal'):
            self.scale = Scale(parent, orient='horizontal', cursor='hand2', from_=from_, to=to,
                               resolution=resolution, label=label, variable=variable, state=state)
            self.scale.grid(row=row, column=column, columnspan=cs, padx=1, pady=0, sticky="nsew")
            self.scale.set(init)
            variable.trace("w", func)

        def get(self):
            return self.scale.get()

        def set(self, value):
            self.scale.set(value)

        def update_length(self, from_, to):
            self.scale.config(from_=from_, to=to)

        def update_state(self, state):
            self.scale.config(state=state)
