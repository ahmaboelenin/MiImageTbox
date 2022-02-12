from tkinter import Canvas
from tkinter.ttk import Frame
from PIL import Image, ImageTk


class ImageBox(Frame):
    def __init__(self, master=None):
        self.selection_rect, self.rect = None, None
        self.max_width = int(master.screen_width * 0.85)
        self.max_height = int(master.screen_height * 0.95)

        self.crop = False

        self.image, self.imageIm, self.imageTk = None, None, None
        self.imScale, self.delta = 1.0, 0.75
        self.center_x, self.center_y = int(self.max_width / 2), int(self.max_height / 2)

        self.actual_size, self.view_size = 0, 0
        self.x_start, self.y_start, self.x_end, self.y_end, self.x_max, self.y_max = 0, 0, 0, 0, 0, 0

        super().__init__(master=master, width=self.max_width, height=self.max_height)
        self.image = self.master.image

        self.imagebox = Canvas(self, width=self.max_width, height=self.max_height, borderwidth=0,
                               highlightthickness=0, bg='#C8C8C8')

        self.imagebox.bind('<Button-1>', lambda event: self.imagebox.scan_mark(event.x, event.y))
        self.imagebox.bind('<Button-3>', self.reset)
        self.imagebox.bind('<B1-Motion>', lambda event: self.imagebox.scan_dragto(event.x, event.y, gain=1))
        self.imagebox.bind('<MouseWheel>', self.wheel)

        self.imagebox.pack()
        # print(self.imagebox.bbox('img'))

    '''____Widget_Functions'''
    def clear_canvas(self):
        self.imagebox.delete("all")

    def wheel(self, event):
        """Zoom with mouse wheel"""
        scale = 1.0
        if event.num == 5 or event.delta == -120:
            scale *= self.delta
            self.imScale *= self.delta
        if event.num == 4 or event.delta == 120:
            scale /= self.delta
            self.imScale /= self.delta

        # Rescale all canvas objects
        x = self.imagebox.canvasx(event.x)
        y = self.imagebox.canvasy(event.y)
        self.imagebox.scale('all', x, y, scale, scale)

        self.clear_canvas()
        height, width = self.image.shape[:2]
        image = Image.fromarray(self.image)
        new_size = int(self.imScale * width), int(self.imScale * height)

        image_tk = ImageTk.PhotoImage(image.resize(new_size))
        self.imagebox.img = image_tk
        self.imagebox.create_image(int(self.max_width / 2), int(self.max_height / 2), image=image_tk, anchor='center')
        self.imagebox.configure(scrollregion=self.imagebox.bbox("all"))

    def reset(self, event=None):
        if self.master.crop:
            self.master.crop = False
            self.disable_bind()
            self.reset_bind()
            self.master.toolBox.enable_elements()
        self.imagebox.delete("all")
        self.imagebox.scan_dragto(0, 0, gain=1)
        self.imagebox.img = self.imageTk
        self.imagebox.create_image(0, 0, image=self.imageTk, anchor='nw', tags='img')
        self.imagebox.configure(cursor="arrow", scrollregion=self.imagebox.bbox("all"))

    def disable_bind(self):
        self.imagebox.unbind('<Button-1>')
        self.imagebox.unbind("<Button-3>")
        self.imagebox.unbind('<B1-Motion>')
        self.imagebox.unbind("<ButtonRelease-1>")
        self.imagebox.unbind('<MouseWheel>')
        self.imagebox.unbind('<Motion>')

    def reset_bind(self):
        self.imagebox.bind('<Button-1>', lambda event: self.imagebox.scan_mark(event.x, event.y))
        self.imagebox.bind('<Button-3>', self.reset)
        self.imagebox.bind('<B1-Motion>', lambda event: self.imagebox.scan_dragto(event.x, event.y, gain=1))
        self.imagebox.bind('<MouseWheel>', self.wheel)

    '''____Image_Functions'''
    def check_image_size(self):
        """This Function Check if Image Size is Good for Screen Resolution.
        if not it Returns the Best Size to Be Shown"""
        width, height = self.imageIm.size[0], self.imageIm.size[1]
        if width > self.max_width or height > self.max_height:
            # Check If Width > Height To Give The Priority To The Width And Get Height Can Be Fit With The New Width.
            if self.master.ratio > 1:
                width = self.max_width
                height = int(width / self.master.ratio)

            # Check If Height > Width To Give The Priority To The Height And Get Width Can Be Fit With The New Height.
            elif self.master.ratio < 1:
                height = self.max_height
                width = int(height * self.master.ratio)
            return width, height
        else:
            return None

    def show_image(self, image=None):
        """This Function:
            1- Check if there is Send Image to get it or get the 'Image'.
            2- Check if Image Shape is Good  With the Screen Resolution or Resize it to Fit the Screen Resolution with
                Keeping the Original Image Shape.
            3- Update Translate Scales with Image Shape.
            4- Check If Image is Gray Scale Image to Disable RGB Image Buttons or RGB Image to Disable Gray Scale Image
                Buttons to Avoid Crashing.
            5- Show Image in the Image-Box."""
        self.imScale, self.delta = 1.0, 0.75
        self.clear_canvas()

        # Check If There Image Is Sent Within The Call To Show It
        if image is None:
            self.image = self.master.image
            self.imageIm = Image.fromarray(self.image)
            self.master.ratio = self.image.shape[1] / self.image.shape[0]
        else:
            self.image = image.copy()
            self.imageIm = Image.fromarray(image)

        # Check If Image Size Is Good For Screen Resolution
        chk = self.check_image_size()
        if chk is not None:
            # If Not Resize It To Fit The Screen Resolution
            self.imageIm = self.imageIm.resize((chk[0], chk[1]))

        self.imageTk = ImageTk.PhotoImage(image=self.imageIm)
        self.imagebox.img = self.imageTk
        self.imagebox.create_image(0, 0, image=self.imageTk, anchor='nw', tags="img")

        self.imagebox.configure(scrollregion=self.imagebox.bbox("all"))

        # Show The Actual Size And The View Size Of The Image.
        self.actual_size, self.view_size = \
            [self.image.shape[1], self.image.shape[0]], [self.imageIm.size[0], self.imageIm.size[1]]
        self.master.toolBox.actual_size_label.config(
            "Actual Size - Width: " + str(self.actual_size[0]) + ", Height: " + str(self.actual_size[1]))
        self.master.toolBox.view_size_label.config(
            "View Size    - Width: " + str(self.view_size[0]) + ", Height: " + str(self.view_size[1]))
        self.master.toolBox.Scales[0].update_length(-abs(self.view_size[0]), self.view_size[0])
        self.master.toolBox.Scales[1].update_length(-abs(self.view_size[1]), self.view_size[1])

    '''____Crop_Functions'''
    def start_crop(self):
        self.reset()
        self.master.crop, self.x_start, self.y_start, self.x_end, self.y_end, self.x_max, self.y_max = \
            True, 0, 0, 0, 0, self.imageTk.width(), self.imageTk.height()
        aa = dict(stipple='gray25', fill='red', dash=(2, 2), outline='')
        self.selection_rect = self.imagebox.create_rectangle(self.x_start, self.y_start, self.x_end, self.y_end,
                                                             dash=(2, 2), fill='', outline='white')
        self.rect = (self.imagebox.create_rectangle(0, 0, self.x_max, self.y_start, **aa),
                     self.imagebox.create_rectangle(0, self.y_start, self.x_start, self.y_end, **aa),
                     self.imagebox.create_rectangle(self.x_end, self.y_start, self.x_max, self.y_end, **aa),
                     self.imagebox.create_rectangle(0, self.y_end, self.x_max, self.y_max, **aa))

        self.imagebox.configure(cursor="crosshair")
        self.imagebox.bind('<Button-1>', self.get_crop_position)
        self.imagebox.bind("<Button-3>", self.reset)
        self.imagebox.bind('<B1-Motion>', self.update_crop_coordinates)
        self.imagebox.bind("<ButtonRelease-1>", self.end_crop)

    def get_crop_position(self, event):
        self.x_start, self.y_start = event.x, event.y

    def update_crop_coordinates(self, event):
        self.x_end, self.y_end = event.x, event.y

        if self.x_start > self.x_end and self.y_start > self.y_end:
            self.imagebox.coords(self.selection_rect, self.x_end, self.y_end, self.x_start, self.y_start)
            self.imagebox.coords(self.rect[0], 0, 0, self.x_max, self.y_end)
            self.imagebox.coords(self.rect[1], 0, self.y_end, self.x_end, self.y_start)
            self.imagebox.coords(self.rect[2], self.x_start, self.y_end, self.x_max, self.y_start)
            self.imagebox.coords(self.rect[3], 0, self.y_start, self.x_max, self.y_max)

        elif self.x_start > self.x_end or self.y_start > self.y_end:
            if self.x_start > self.x_end:
                self.imagebox.coords(self.selection_rect, self.x_end, self.y_start, self.x_start, self.y_end)
                self.imagebox.coords(self.rect[0], 0, 0, self.x_max, self.y_start)
                self.imagebox.coords(self.rect[1], 0, self.y_start, self.x_end, self.y_end)
                self.imagebox.coords(self.rect[2], self.x_start, self.y_start, self.x_max, self.y_end)
                self.imagebox.coords(self.rect[3], 0, self.y_end, self.x_max, self.y_max)
            if self.y_start > self.y_end:
                self.imagebox.coords(self.selection_rect, self.x_start, self.y_end, self.x_end, self.y_start)
                self.imagebox.coords(self.rect[0], 0, 0, self.x_max, self.y_end)
                self.imagebox.coords(self.rect[1], 0, self.y_end, self.x_start, self.y_start)
                self.imagebox.coords(self.rect[2], self.x_end, self.y_end, self.x_max, self.y_start)
                self.imagebox.coords(self.rect[3], 0, self.y_start, self.x_max, self.y_max)
        else:
            self.imagebox.coords(self.selection_rect, self.x_start, self.y_start, self.x_end, self.y_end)
            self.imagebox.coords(self.rect[0], 0, 0, self.x_max, self.y_start)
            self.imagebox.coords(self.rect[1], 0, self.y_start, self.x_start, self.y_end)
            self.imagebox.coords(self.rect[2], self.x_end, self.y_start, self.x_max, self.y_end)
            self.imagebox.coords(self.rect[3], 0, self.y_end, self.x_max, self.y_max)

    def end_crop(self, event):
        self.reset()
        width_ratio = self.actual_size[0] / self.view_size[0]
        height_ratio = self.actual_size[1] / self.view_size[1]
        if width_ratio != 1 or height_ratio != 1:
            self.x_start, self.x_end = int(self.x_start * width_ratio), int(self.x_end * width_ratio)
            self.y_start, self.y_end = int(self.y_start * height_ratio), int(self.y_end * height_ratio)
        self.master.toolBox.end_crop(self.x_start, self.y_start, self.x_end, self.y_end)

    '''____Skew_Functions'''
    def start_skew(self):
        self.reset()
        self.master.crop, self.x_start, self.y_start, self.x_end, self.y_end, self.x_max, self.y_max = \
            True, 0, 0, 0, 0, self.imageTk.width(), self.imageTk.height()

        self.imagebox.configure(cursor="target")
        self.imagebox.bind('<Button-1>', self.get_skew_position)
        self.imagebox.bind("<Button-3>", self.reset)

    def get_skew_position(self, event):
        x, y = event.x, event.y
        self.reset()
        width_ratio = self.actual_size[0] / self.view_size[0]
        height_ratio = self.actual_size[1] / self.view_size[1]
        if width_ratio != 1 or height_ratio != 1:
            self.x_start, self.x_end = int(self.x_start * width_ratio), int(self.x_end * width_ratio)
            self.y_start, self.y_end = int(self.y_start * height_ratio), int(self.y_end * height_ratio)
        self.master.toolBox.end_skew(x, y)

    '''____Four_Points_Transform_Functions'''
    def start_four_points_transform(self):
        self.reset()
        self.master.crop, self.rect = True, []
        self.disable_bind()
        self.imagebox.configure(cursor="target")
        self.imagebox.bind('<Button-1>', self.get_mouse_position2)
        self.imagebox.bind("<Button-3>", self.reset)

    def get_mouse_position2(self, event):
        x, y = event.x, event.y
        self.imagebox.create_oval(x - 4, y - 4, x + 4, y + 4, fill="black", outline="", width=4)
        if len(self.rect) < 4:
            self.rect.append([x, y])
        else:
            self.reset()
            self.end_four_points_transform()

        if len(self.rect) != 1:
            a = self.rect[len(self.rect)-2]
            b = self.rect[len(self.rect) - 1]
            self.imagebox.create_line(a[0], a[1], b[0], b[1])

            if len(self.rect) == 4:
                a = self.rect[len(self.rect) - 1]
                b = self.rect[0]
                self.imagebox.create_line(a[0], a[1], b[0], b[1])

    def end_four_points_transform(self):
        self.reset()
        width_ratio = self.actual_size[0] / self.view_size[0]
        height_ratio = self.actual_size[1] / self.view_size[1]
        if width_ratio != 1 or height_ratio != 1:
            for i in range(len(self.rect)):
                self.rect[i][0] = int(self.rect[i][0] * height_ratio)
                self.rect[i][1] = int(self.rect[i][1] * width_ratio)
        self.master.toolBox.end_four_points_transform(self.rect)

    '''____Zoom_Functions'''
    def start_zoom(self):
        self.reset()
        self.master.crop = True
        self.disable_bind()
        self.imagebox.bind('<Button-1>', self.get_mouse_position3)
        self.imagebox.bind('<Motion>', self.update_zoom_coordinates)
        self.imagebox.bind("<Button-3>", self.reset)
        self.imagebox.configure(cursor="target")

        self.selection_rect = self.imagebox.create_rectangle(0, 0, 0, 0, dash=(2, 2), fill='', outline='white')

    def update_zoom_coordinates(self, event):
        x, y = event.x, event.y
        width, height = self.view_size[0]/4, self.view_size[1]/4
        x_start, y_start, x_end, y_end = x - width, y - height, x + width, y + height
        self.imagebox.coords(self.selection_rect, x_start, y_start, x_end, y_end)

    def get_mouse_position3(self, event):
        x, y = event.x, event.y
        self.reset()
        width_ratio = self.actual_size[0] / self.view_size[0]
        height_ratio = self.actual_size[1] / self.view_size[1]
        if width_ratio != 1 or height_ratio != 1:
            x, y = int(x * width_ratio), int(y * width_ratio)
        self.master.toolBox.end_zoom(x, y)
