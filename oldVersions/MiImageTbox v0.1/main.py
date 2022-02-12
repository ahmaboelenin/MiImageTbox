import sys
from tkinter import Tk, Canvas, Scale, Menu, PhotoImage, filedialog, IntVar, DoubleVar, messagebox, simpledialog
from tkinter.ttk import Frame, Scrollbar, Button, Label, Entry
import imageScripts as im
from PIL import Image, ImageTk

'''____Declarations____'''
global image_box, image, image_temp, imagetk, actual_size, view_size, max_width, max_height, actual_size_label, \
    view_size_label, main_buttons, transformation_buttons, filter_buttons, gray_buttons, rgb_buttons, scales, \
    entry_width, entry_height, selection_rect, rect

screen_width, screen_height = 0, 0  # Images Size Variables
image2, ratio = None, 0  # Image Variables
image_history, index = [], -1  # History and Index For Undo, Redo
x_start, y_start, x_end, y_end, x_max, y_max = 0, 0, 0, 0, 0, 0  # Crop Variables
register, un_do, re_do, crop = True, False, False, False  # Controllers

'''____Image_Functions____'''


def open_image(path=None):
    """This Function Loads Image"""
    global image, ratio, image_history, index
    if path is None:
        path = 'img/sample.jpg'  # Load The Sample Image
    temp_image = im.image_read(path)
    if temp_image is not None:
        image_history, index = [], -1
        image = temp_image
        ratio = image.shape[1] / image.shape[0]  # shape[1] Return Rows, shape[0] Return Columns
        register_image()


def swap_image(img):
    """This Function Make the Edited Image as the Primary to Show it."""
    global image
    if img is None:  # or np.array_equal(img, image)
        return
    image = img
    register_image()


def register_image():
    """This Function Make History for Editing Image for Undo any Changes After Doing it."""
    global index, un_do
    if register:
        if un_do:
            un_do = False
            main_buttons[5].config("disabled")
            for i in range(index + 1, len(image_history)):
                image_history.pop(-1)
        index += 1
        if index > 0:
            main_buttons[3].config("normal")
            main_buttons[4].config("normal")
        else:
            main_buttons[3].config("disabled")
            main_buttons[4].config("disabled")
        image_history.append(image)
    show_image()
    check_image_channels()


def show_image(img=None):
    """ This Function:
    1- Check if there is Send Image to get it or get the 'Image'.
    2- Check if Image Shape is Good  With the Screen Resolution or Resize it to Fit the Screen Resolution with Keeping
        the Original Image Shape.
    3- Update Translate Scales with Image Shape.
    4- Check If Image is Gray Scale Image to Disable RGB Image Buttons or RGB Image to Disable Gray Scale Image Buttons
        to Avoid Crashing.
    5- Show Image in the Image-Box."""
    global imagetk, ratio, actual_size, view_size

    if img is None:  # Check If There Image Is Sent Within The Call To Show It
        temp = Image.fromarray(image)
        ratio = image.shape[1] / image.shape[0]
    else:
        temp = Image.fromarray(img)

    chk = check_image_size(temp)  # Check If Image Size Is Good For Screen Resolution
    if chk is not None:
        temp = temp.resize((chk[0], chk[1]))  # If Not Resize It To Fit The Screen Resolution

    imagetk = ImageTk.PhotoImage(image=temp)  # Read Image As TK Photo To Show It In Tkinter Canvas
    image_box.img = imagetk
    image_box.create_image(0, 0, image=imagetk, anchor='nw')
    # image_box.create_image(int(max_width / 2), int(max_height / 2), image=imagetk, anchor='center')

    # Show The Actual Size And The View Size Of The Image.
    actual_size, view_size = [image.shape[1], image.shape[0]], [temp.size[0], temp.size[1]]
    actual_size_label.config("Actual Size - Width: " + str(actual_size[0]) + ", Height: " + str(actual_size[1]))
    view_size_label.config("   View Size - Width: " + str(view_size[0]) + ", Height: " + str(view_size[1]))
    scales[0].update_length(-abs(view_size[0]), view_size[0])  # Update Translate Scales with Image Shape
    scales[1].update_length(-abs(view_size[1]), view_size[1])


def check_image_size(img):
    """This Function Check if Image Size is Good for Screen Resolution. if not it Returns the Best Size to Be Shown."""
    global max_width, max_height
    max_width, max_height, width, height = int(screen_width * 0.85), int(screen_height * 0.95), img.size[0], img.size[1]
    if width > max_width or height > max_height:
        # Check If Width > Height To Give The Priority To The Width And Get Height Can Be Fit With The New Width.
        if ratio > 1:
            width = max_width
            height = int(width / ratio)
        # Check If Height > Width To Give The Priority To The Height And Get Width Can Be Fit With The New Height.
        elif ratio < 1:
            height = max_height
            width = int(height * ratio)
        return width, height
    else:
        return None


def check_image_channels(img=None):
    """This Function Checks If Image is Gray Scale Image to Disable RGB Image Buttons or RGB Image to Disable Gray Scale
    Image Buttons to Avoid Crashing."""
    if img is None:
        img = image
    if len(img.shape) == 3:  # Check If Image is RGB to Disable Gray Buttons to Avoid Crashing
        rgb_buttons[0].config("disabled")
        for btn in rgb_buttons[1:]:
            btn.config("normal")
        gray_buttons[0].config("normal")
        for btn in gray_buttons[1:]:
            btn.config("disabled")
        for scale in scales[7:10]:
            scale.update_state("normal")
    else:  # Check If Image is Gray to Disable RGB Buttons to Avoid Crashing
        rgb_buttons[0].config("normal")
        for btn in rgb_buttons[1:]:
            btn.config("disabled")
        gray_buttons[0].config("disabled")
        for btn in gray_buttons[1:]:
            btn.config("normal")
        for scale in scales[7:10]:
            scale.update_state("disabled")


def browse_files():
    """This Function Open File Dialog to Read the New Image."""
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File",
                                          filetypes=(("Image files", "*.jpg *.jpeg *png"), ("all files", "*.*")))
    return filename


'''____Essential_Functions____'''


def is_number(num):
    """This Function Check if the Entry is Number or Not to Avoid Crashing."""
    try:
        return int(num)
    except ValueError:
        return False


def reset(event=None):
    global register, crop
    if crop:
        crop = False
        image_box.unbind('<Button-1>')
        image_box.unbind("<Button-3>")
        image_box.unbind('<B1-Motion>')
        image_box.unbind("<ButtonRelease-1>")
        image_box.configure(cursor="arrow")
        image_box.delete("all")
        show_image()
    reset_scales()
    enable_buttons()
    register = True


def get_height(height):
    if height is None:
        width = is_number(entry_width.get())
        height = int(width / ratio)
    entry_height.reset()
    entry_height.set(height)
    return height


def get_width(width):
    if width is None:
        height = is_number(entry_height.get())
        width = int(height * ratio)
    entry_width.reset()
    entry_width.set(width)
    return width


def disable_buttons(case=None):
    for btn in main_buttons + transformation_buttons + filter_buttons + gray_buttons + rgb_buttons:
        btn.config("disabled")
    if case == 0:
        main_buttons[3].config("normal")
    else:
        main_buttons[6].config("normal")
        main_buttons[7].config("normal")


def enable_buttons():
    for btn in main_buttons + transformation_buttons + filter_buttons:
        btn.config("normal")
    main_buttons[4].config("disabled")
    main_buttons[5].config("disabled")
    main_buttons[6].config("disabled")
    main_buttons[7].config("disabled")
    check_image_channels()
    check_undo_redo_buttons()


def check_undo_redo_buttons():
    if len(image_history) > 1:
        main_buttons[4].config("normal")    # Undo Button
    if index < (len(image_history)-1):
        main_buttons[5].config("normal")    # Redo Button


def reset_scales():
    for i in range(len(scales)):
        scales[i].update_state("normal")
        scales[i].set(0)  # Translate-X, Translate-Y, Blending, Canny, RGB
        if i == 2:
            scales[i].update_state(state="disabled")
        elif i == 3:  # Brightness
            scales[i].set(255)
        elif i == 4:
            scales[i].set(127)  # Contrast
        elif i == 5:
            scales[i].set(1)  # Gamma


def disable_scales(case=None):
    if case is None:
        for scale in scales:
            scale.update_state('disabled')
    elif case == 0:
        for scale in scales[2:]:  # Translate-X, Translate-Y
            scale.update_state('disabled')
    elif case == 1:  # Blending
        for scale in scales[:2] + scales[3:]:
            scale.update_state('disabled')
    elif case == 2:  # Brightness and Contrast
        for scale in scales[:3] + scales[5:]:
            scale.update_state('disabled')
    elif case == 3:  # Gamma
        for scale in scales[:5] + scales[6:]:
            scale.update_state('disabled')
    elif case == 4:  # Canny
        for scale in scales[:6] + scales[7:]:
            scale.update_state('disabled')
    elif case == 5:
        for scale in scales[:7]:
            scale.update_state('disabled')
        scales[10].update_state('disabled')
    elif case == 6:
        for scale in scales[:10]:
            scale.update_state('disabled')
        scales[11].update_state('disabled')
    elif case == 7:
        for scale in scales[:11]:
            scale.update_state('disabled')


def rotate_with_angle(angle):
    temp_image = im.image_rotate(image, angle)
    swap_image(temp_image)


def force_resize(width, height):
    temp_image = im.image_resize(image, None, width, height)
    swap_image(temp_image)


class ToolBox(Frame):
    def __init__(self):
        global actual_size_label, view_size_label, entry_width, entry_height, main_buttons, transformation_buttons,\
            filter_buttons, gray_buttons, rgb_buttons, scales
        translateX_var, translateY_var, brightness_var, contrast_var, canny_var, red_var, green_var, blue_var, obj_var,\
            thresh_var, gamma_var, blind_var = IntVar(), IntVar(), IntVar(), IntVar(), IntVar(), IntVar(), IntVar(), \
            IntVar(), IntVar(), IntVar(), DoubleVar(), DoubleVar()

        super().__init__()

        def _bound_to_mousewheel(event):
            tool_box_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbound_to_mousewheel(event):
            tool_box_canvas.unbind_all("<MouseWheel>")

        def _on_mousewheel(event):  # ScrollBar Movement with Mouse Wheel
            tool_box_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        '''____Main_Buttons_Functions____'''

        def open_button_func():
            filename = browse_files()
            if filename is None:
                return
            open_image(filename)

        def save_button_func():
            filename = filedialog.asksaveasfilename(initialfile='Untitled.jpg', defaultextension=".jpg",
                                                    filetypes=[("jpg Image", "*.jpg"), ("jpeg Image", "*.jpeg"),
                                                               ("All Files", "*.*")])
            try:
                im.image_save(image, filename)
            except:
                return

        def undo_button_func():
            global image, index, un_do
            un_do, index = True, index - 1
            image = image_history[index]
            if index == 0:
                main_buttons[4].config("disabled")
            main_buttons[5].config("normal")
            show_image()
            check_image_channels()

        def redo_button_func():
            global image, index, re_do
            re_do, index = True, index + 1
            image = image_history[index]
            if index == len(image_history) - 1:
                main_buttons[5].config("disabled")
            main_buttons[4].config("normal")
            show_image()
            check_image_channels()

        def done_button_func():
            temp = image_temp
            reset()
            swap_image(temp)

        def cancel_button_func():
            reset()
            show_image()
            check_image_channels()

        def show_original_button_func():
            if main_buttons[3].get_text() == "Show Original":
                main_buttons[3].set_text("Show Current")
                show_image(image_history[0])
                disable_buttons(0)
                disable_scales()
            else:
                main_buttons[3].set_text("Show Original")
                show_image(image_history[-1])
                reset_scales()
                reset()
                check_image_channels()

        def load_image_button_func():
            global image2
            filename = browse_files()
            temp_image = im.image_read(filename)
            if temp_image is not None:
                scales[2].update_state(state="normal")
                image2 = temp_image

        '''____Transformation_Buttons_Functions____'''

        def rotate_button_func(angle):  # Rotate Button Function
            temp_image = im.image_rotate(image, angle)
            swap_image(temp_image)

        def flip_button_func(case):  # Flip Button Function
            temp_image = im.image_flip(image, case)
            swap_image(temp_image)

        def resize_button_func():
            width, height = is_number(entry_width.get()), is_number(entry_height.get())
            if (width is False and height is False) or (width == 0 or height == 0):
                return
            if width is False:
                width = get_width(None)
            if height is False or width / height != ratio:
                height = get_height(None)
            temp_image = im.image_resize(image, None, width, height)
            swap_image(temp_image)

        def start_crop():
            global selection_rect, rect, x_start, y_start, x_end, y_end, x_max, y_max, crop
            x_start, y_start, x_end, y_end, x_max, y_max, crop = 0, 0, 0, 0, imagetk.width(), imagetk.height(), True
            aa = dict(stipple='gray25', fill='red', dash=(2, 2), outline='')
            selection_rect = image_box.create_rectangle(x_start, y_start, x_end, y_end, dash=(2, 2), fill='',
                                                        outline='white')
            rect = (image_box.create_rectangle(0, 0, x_max, y_start, **aa),
                    image_box.create_rectangle(0, y_start, x_start, y_end, **aa),
                    image_box.create_rectangle(x_end, y_start, x_max, y_end, **aa),
                    image_box.create_rectangle(0, y_end, x_max, y_max, **aa))
            image_box.bind('<Button-1>', get_crop_position)
            image_box.bind("<Button-3>", reset)
            image_box.bind('<B1-Motion>', update_crop_coordinates)
            image_box.bind("<ButtonRelease-1>", end_crop)
            image_box.configure(cursor="crosshair")
            disable_buttons()
            main_buttons[6].config("disabled")

        def get_crop_position(event):
            """This Function Gets the Current Mouse Position."""
            global x_start, y_start
            x_start, y_start = event.x, event.y

        def update_crop_coordinates(event):
            global x_end, y_end
            x_end, y_end = event.x, event.y

            if x_start > x_end and y_start > y_end:
                image_box.coords(selection_rect, x_end, y_end, x_start, y_start)
                image_box.coords(rect[0], 0, 0, x_max, y_end)
                image_box.coords(rect[1], 0, y_end, x_end, y_start)
                image_box.coords(rect[2], x_start, y_end, x_max, y_start)
                image_box.coords(rect[3], 0, y_start, x_max, y_max)

            elif x_start > x_end or y_start > y_end:
                if x_start > x_end:
                    image_box.coords(selection_rect, x_end, y_start, x_start, y_end)
                    image_box.coords(rect[0], 0, 0, x_max, y_start)
                    image_box.coords(rect[1], 0, y_start, x_end, y_end)
                    image_box.coords(rect[2], x_start, y_start, x_max, y_end)
                    image_box.coords(rect[3], 0, y_end, x_max, y_max)
                if y_start > y_end:
                    image_box.coords(selection_rect, x_start, y_end, x_end, y_start)
                    image_box.coords(rect[0], 0, 0, x_max, y_end)
                    image_box.coords(rect[1], 0, y_end, x_start, y_start)
                    image_box.coords(rect[2], x_end, y_end, x_max, y_start)
                    image_box.coords(rect[3], 0, y_start, x_max, y_max)
            else:
                image_box.coords(selection_rect, x_start, y_start, x_end, y_end)
                image_box.coords(rect[0], 0, 0, x_max, y_start)
                image_box.coords(rect[1], 0, y_start, x_start, y_end)
                image_box.coords(rect[2], x_end, y_start, x_max, y_end)
                image_box.coords(rect[3], 0, y_end, x_max, y_max)

        def end_crop(event):
            global x_start, y_start, x_end, y_end
            reset()
            width_ratio = actual_size[0] / view_size[0]
            height_ratio = actual_size[1] / view_size[1]
            if width_ratio != 1 or height_ratio != 1:
                x_start, x_end = int(x_start * width_ratio), int(x_end * width_ratio)
                y_start, y_end = int(y_start * height_ratio), int(y_end * height_ratio)
            temp_image = im.image_crop(image, x_start, y_start, x_end, y_end)
            swap_image(temp_image)

        def start_4points_transform():
            global crop, rect
            crop, rect = True, []
            image_box.bind('<Button-1>', get_4points_position)
            image_box.bind("<Button-3>", reset)
            image_box.configure(cursor="target")
            disable_buttons()
            main_buttons[6].config("disabled")

        def get_4points_position(event):
            x, y = event.x, event.y
            image_box.create_oval(x - 4, y - 4, x + 4, y + 4, fill="black", outline="", width=4)
            rect.append([x, y])

            if len(rect) == 4:
                end_4points_transform()

        def end_4points_transform():
            reset()
            width_ratio = actual_size[0] / view_size[0]
            height_ratio = actual_size[1] / view_size[1]
            if width_ratio != 1 or height_ratio != 1:
                for i in range(len(rect)):
                    rect[i][0] = int(rect[i][0] * height_ratio)
                    rect[i][1] = int(rect[i][1] * width_ratio)

            temp_image = im.image_four_points_transform(image, rect)
            swap_image(temp_image)
            rect.clear()

        def start_zoom():
            global crop
            crop = True
            image_box.bind('<Button-1>', get_zoom_position)
            image_box.bind("<Button-3>", reset)
            image_box.configure(cursor="target")
            disable_buttons()
            main_buttons[6].config("disabled")

        def get_zoom_position(event):
            x, y = event.x, event.y
            reset()
            temp_image = im.image_zoom(image, x, y)
            swap_image(temp_image)

        def un_translate_button_func():
            temp_image = im.image_un_translate(image)
            swap_image(temp_image)

        def skewing_button_func():
            temp_image = im.image_skewing(image)
            swap_image(temp_image)

        def un_skewing_button_func():
            temp_image = im.image_un_skewing(image)
            swap_image(temp_image)

        def edge_detection_func():
            temp_image = im.edge_detection(image)
            swap_image(temp_image)

        '''____Enhancement_Buttons_Functions____'''

        def adjust_gamma_button_func():
            temp_image = im.adjust_gamma(image)
            swap_image(temp_image)

        def thresh_button_func():
            mode_ = thresh_lbl.get_text()
            if mode_ == 'binary':
                thresh_lbl.config('binary_inv')
            elif mode_ == 'binary_inv':
                thresh_lbl.config('to_zero')
            elif mode_ == 'to_zero':
                thresh_lbl.config('to_zero_inv')
            elif mode_ == 'to_zero_inv':
                thresh_lbl.config('trunc')
            elif mode_ == 'trunc':
                thresh_lbl.config('binary')
            if not register:
                thresh_scale_func()

        def filter_func(filter_name):
            temp_image = image.copy()
            temp_image = im.image_filters(temp_image, filter_name)
            swap_image(temp_image)

        def gray_filter_func(filter_name):
            temp_image = image.copy()
            temp_image = im.gray_image_filters(temp_image, filter_name)
            swap_image(temp_image)

        def rgb_filter_func(filter_name):
            temp_image = image.copy()
            temp_image = im.rgb_image_filters(temp_image, filter_name)
            swap_image(temp_image)

        def rgb_bgr_func():
            if rgb_buttons[1].get_text() == "RGB To BGR":
                rgb_buttons[1].set_text("BGR To RGB")
                temp_image = im.rgb2bgr(image)
            else:
                rgb_buttons[1].set_text("RGB To BGR")
                temp_image = im.bgr2rgb(image)
            swap_image(temp_image)

        '''____Scales_Functions____'''

        def translate_scale_func(*args):
            global register, image_temp
            if register:
                register = False
                disable_buttons()
                disable_scales(0)

            translate_x, translate_y = translateX_var.get(), translateY_var.get()
            image_temp = im.image_translate(image, translate_x, translate_y)
            show_image(image_temp)

        def blending_scale_func(*args):
            global register, image_temp
            if register:
                register = False
                disable_buttons()
                disable_scales(1)

            alpha = blind_var.get()
            beta = 1 - alpha
            image_temp = im.image_blending(image, beta, image2, alpha)
            show_image(image_temp)

        def brightness_contrast_scale_func(*args):
            global register, image_temp
            if register:
                register = False
                disable_buttons()
                disable_scales(2)

            brightness, contrast = brightness_var.get(), contrast_var.get()
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
                image_temp = im.image_addweighted(image, alpha, image, gamma)
            else:
                image_temp = image

            if contrast != 0:
                alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
                gamma = 127 * (1 - alpha)
                image_temp = im.image_addweighted(image_temp, alpha, image_temp, gamma)
            if brightness == 0 and contrast == 0:
                return
            show_image(image_temp)

        def gamma_scale_func(*args):
            global register, image_temp
            if register:
                register = False
                disable_buttons()
                disable_scales(3)

            alpha = gamma_var.get()
            image_temp = im.adjust_gamma(image, alpha)
            show_image(image_temp)

        def canny_scale_func(*args):
            global register, image_temp
            if register:
                register = False
                disable_buttons()
                disable_scales(4)

            alpha = canny_var.get()
            beta = alpha * 2
            image_temp = im.image_canny_edge(image, alpha, beta)
            show_image(image_temp)

        def rgb_scale_func(*args):
            global register, image_temp
            if register:
                register = False
                disable_buttons()
                disable_scales(5)

            alpha_r, alpha_g, alpha_b = red_var.get(), green_var.get(), blue_var.get()
            image_temp = im.rgb_change(image, alpha_r, alpha_g, alpha_b)
            show_image(image_temp)

        def obj_scale_func(*args):
            global register, image_temp
            if register:
                register = False
                disable_buttons()
                disable_scales(6)

            obj = obj_var.get()
            if obj != 0:
                image_temp = im.k_means_edge_detection(image, obj)
                show_image(image_temp)

        def thresh_scale_func(*args):
            global register, image_temp
            if register:
                register = False
                disable_buttons()
                disable_scales(7)
                filter_buttons[0].config("normal")

            value, mode = thresh_var.get(), thresh_lbl.get_text()
            image_temp = im.thresh(image, mode, value)
            show_image(image_temp)

        '''__________________________________________________________________________________________________________'''
        tool_box_canvas = Canvas(self, borderwidth=0, highlightthickness=0, width=int(screen_width * 0.136))
        scrollbar = Scrollbar(self, orient='vertical', command=tool_box_canvas.yview)

        tool_box = Frame(tool_box_canvas)
        '''____Tool_Box_Elements____'''
        actual_size_label = self.Label(tool_box, "Actual Size: ", 0, 0, 2)
        view_size_label = self.Label(tool_box, "View Size: ", 1, 0, 2)
        self.Label(tool_box, "-------------------------------------------------", 4, 0, 2)

        '''____Buttons____'''
        main_buttons = [
            self.Button(tool_box, "New Image", lambda x: open_button_func(), "", 5, 0, 1),
            self.Button(tool_box, "Save Image", lambda x: save_button_func(), "", 5, 1, 1),
            self.Button(tool_box, "Load Blending Image", lambda x: load_image_button_func(), "", 6, 0, 1),
            self.Button(tool_box, "Show Original", lambda x: show_original_button_func(), "", 6, 1, 1,
                        "disabled"),
            self.Button(tool_box, "Undo", lambda x: undo_button_func(), "", 7, 0, 1, "disabled"),
            self.Button(tool_box, "Redo", lambda x: redo_button_func(), "", 7, 1, 1, "disabled"),
            self.Button(tool_box, "Done", lambda x: done_button_func(), "", 8, 0, 1, "disabled"),
            self.Button(tool_box, "Cancel", lambda x: cancel_button_func(), "", 8, 1, 1, "disabled"),
            self.Button(tool_box, "Get Histogram", lambda x: im.draw_histogram(image), "", 9, 0, 2),
            self.Button(tool_box, "Resize", lambda x: resize_button_func(), "None", 13, 0, 2)]

        transformation_buttons = [
            self.Button(tool_box, "Rotate Left", lambda x: rotate_button_func(90), "", 15, 0, 1),
            self.Button(tool_box, "Rotate Right", lambda x: rotate_button_func(-90), "", 15, 1, 1),
            self.Button(tool_box, "Vertical Flip", lambda x: flip_button_func(1), "", 16, 0, 1),
            self.Button(tool_box, "Horizontal Flip", lambda x: flip_button_func(0), "", 16, 1, 1),
            self.Button(tool_box, "Rotate 180° Or Flip XY", lambda x: rotate_button_func(180), "", 17, 0, 2),
            self.Button(tool_box, "Crop", lambda x: start_crop(), "", 19, 0, 1),
            self.Button(tool_box, "4 Points Transform", lambda x: start_4points_transform(), "", 19, 1, 1),
            self.Button(tool_box, "Zoom", lambda x: start_zoom(), "", 20, 0, 1),
            self.Button(tool_box, "Un-Translate", lambda x: un_translate_button_func(), "", 20, 1, 1),
            self.Button(tool_box, "Skewing", lambda x: skewing_button_func(), "", 21, 0, 1),
            self.Button(tool_box, "Un-Skewing", lambda x: un_skewing_button_func(), "", 21, 1, 1)]

        filter_buttons = [
            self.Button(tool_box, "Thresholding Mode", lambda x: thresh_button_func(), "", 34, 0, 2),
            self.Button(tool_box, "Adjust Gamma", lambda x: adjust_gamma_button_func(), "", 40, 0, 1),
            self.Button(tool_box, "Salt & Pepper Noise", lambda x: filter_func("salt_pepper"), "", 40, 1, 1),
            self.Button(tool_box, "Black And White", lambda x: filter_func("black_white"), "", 41, 0, 1),
            self.Button(tool_box, "Negative", lambda x: filter_func("negative"), "", 41, 1, 1),
            self.Button(tool_box, "Blurring", lambda x: filter_func("blur"), "", 42, 0, 1),
            self.Button(tool_box, "Median Blurring", lambda x: filter_func("median"), "", 42, 1, 1),
            self.Button(tool_box, "Gaussian Filter", lambda x: filter_func("gaussian"), "", 43, 0, 1),
            self.Button(tool_box, "Pyramidal Filter", lambda x: filter_func("pyramidal"), "", 43, 1, 1),
            self.Button(tool_box, "Circular Filter", lambda x: filter_func("circular"), "", 44, 0, 1),
            self.Button(tool_box, "Cone Filter", lambda x: filter_func("cone"), "", 44, 1, 1),
            self.Button(tool_box, "Bilateral", lambda x: filter_func("bilateral"), "", 45, 0, 1),
            self.Button(tool_box, "Emboss Filter", lambda x: filter_func("emboss"), "", 45, 1, 1),
            self.Button(tool_box, "Sharpen Filter", lambda x: filter_func("sharpen"), "", 46, 0, 1),
            self.Button(tool_box, "Add Border", lambda x: filter_func("border"), "", 46, 1, 1),
            self.Button(tool_box, "Laplace Edge", lambda x: filter_func("laplace_edge"), "", 47, 0, 1),
            self.Button(tool_box, "Outline Edge", lambda x: filter_func("outline_edge"), "", 47, 1, 1),
            self.Button(tool_box, "Sobel Y - 8U", lambda x: filter_func("sobelY"), "", 48, 0, 1),
            self.Button(tool_box, "Sobel Y - 8U(64F)", lambda x: filter_func("sobelY_64f"), "", 48, 1, 1),
            self.Button(tool_box, "Sobel X - 8U", lambda x: filter_func("sobelX"), "", 49, 0, 1),
            self.Button(tool_box, "Sobel X - 8U(64F)", lambda x: filter_func("sobelX_64f"), "", 49, 1, 1),
            self.Button(tool_box, "Sobel Y, X - 8U", lambda x: filter_func("sobelYX"), "", 50, 0, 1),
            self.Button(tool_box, "Sobel Y, X - 8U(64F)", lambda x: filter_func("sobelYX_64f"), "", 50, 1, 1),
            self.Button(tool_box, "Scharr Y", lambda x: filter_func("scharrY"), "", 51, 0, 1),
            self.Button(tool_box, "Scharr X", lambda x: filter_func("scharrX"), "", 51, 1, 1),
            self.Button(tool_box, "Prewitt Y", lambda x: filter_func("prewittY"), "", 52, 0, 1),
            self.Button(tool_box, "Prewitt X", lambda x: filter_func("prewittX"), "", 52, 1, 1),
            self.Button(tool_box, "Scharr Y, X", lambda x: filter_func("scharrYX"), "", 53, 0, 1),
            self.Button(tool_box, "Prewitt Y, X", lambda x: filter_func("prewittYX"), "", 53, 1, 1),
            self.Button(tool_box, "Edge Detection", lambda x: edge_detection_func(), "", 54, 0, 2)]

        gray_buttons = [
            self.Button(tool_box, "Gray Scale", lambda x: gray_filter_func("gray_scale"), "", 60, 0, 1),
            self.Button(tool_box, "Equalize Histogram", lambda x: gray_filter_func("hist_equalize"), "", 60, 1, 1),
            self.Button(tool_box, "Log Transformation", lambda x: gray_filter_func("log_trans"), "", 61, 0, 1),
            self.Button(tool_box, "Bit Plane Slicing", lambda x: gray_filter_func("bit_plane_slicing"), "", 61, 1, 1),
            self.Button(tool_box, "Gray Level Slicing", lambda x: gray_filter_func("gray_level_slicing"), "", 62, 0, 1),
            self.Button(tool_box, "Thresh", lambda x: gray_filter_func("thresh"), "", 62, 1, 1)]

        rgb_buttons = [
            self.Button(tool_box, "Gray To BGR", lambda x: rgb_filter_func("gray2bgr"), "", 73, 0, 1),
            self.Button(tool_box, "RGB To BGR", lambda x: rgb_bgr_func(), "", 73, 1, 1),
            self.Button(tool_box, "CLAHE", lambda x: rgb_filter_func("clahe"), "", 74, 0, 1),
            self.Button(tool_box, "HDR Effect", lambda x: rgb_filter_func("hdr"), "", 74, 1, 1),
            self.Button(tool_box, "Summer Effect", lambda x: rgb_filter_func("summer"), "", 75, 0, 1),
            self.Button(tool_box, "Winter Effect", lambda x: rgb_filter_func("winter"), "", 75, 1, 1),
            self.Button(tool_box, "Sepia Filter", lambda x: rgb_filter_func("sepia"), "", 76, 0, 1),
            self.Button(tool_box, "Edge Mask", lambda x: rgb_filter_func("edge_mask"), "", 76, 1, 1),
            self.Button(tool_box, "Pencil Gary Sketch", lambda x: rgb_filter_func("pencil_gray"), "", 77, 0, 1),
            self.Button(tool_box, "Pencil RGB Sketch", lambda x: rgb_filter_func("pencil_rgb"), "", 77, 1, 1),
            self.Button(tool_box, "Color Quantization", lambda x: rgb_filter_func("color_quantization"), "", 78, 0, 1),
            self.Button(tool_box, "Cartoon", lambda x: rgb_filter_func("cartoon"), "", 78, 1, 1)]

        '''____Entries____'''
        entry_width = self.Entry(tool_box, "Width", lambda x: get_height(None), 12, 0)
        entry_height = self.Entry(tool_box, "Height", lambda x: get_width(None), 12, 1)

        '''____Scales____'''
        # Scale(parent, from_, to, resolution, label, variable, func, row, column, cs, ini)
        scales = [
            self.Scale(tool_box, -100, 100, 10, 'Translate-X', translateX_var, translate_scale_func, 23, 0, 1, 0),
            self.Scale(tool_box, -100, 100, 10, 'Translate-Y', translateY_var, translate_scale_func, 23, 1, 1, 0),
            self.Scale(tool_box, 0, 1, 0.1, 'Blending', blind_var, blending_scale_func, 25, 0, 2, 0),
            self.Scale(tool_box, 0, 510, 1, 'Brightness', brightness_var, brightness_contrast_scale_func, 30, 0, 1,
                       255),
            self.Scale(tool_box, 0, 254, 1, 'Contrast', contrast_var, brightness_contrast_scale_func, 30, 1, 1, 127),
            self.Scale(tool_box, 0.1, 2, 0.1, 'Gamma', gamma_var, gamma_scale_func, 32, 0, 1, 1),
            self.Scale(tool_box, 0, 255, 8, 'Canny Edge', canny_var, canny_scale_func, 32, 1, 1, 0),
            self.Scale(tool_box, 0, 255, 1, 'Red', red_var, rgb_scale_func, 70, 0, 1, 0),
            self.Scale(tool_box, 0, 255, 1, 'Green', green_var, rgb_scale_func, 70, 1, 1, 0),
            self.Scale(tool_box, 0, 255, 1, 'Blue', blue_var, rgb_scale_func, 71, 0, 1, 0),
            self.Scale(tool_box, 0, 20, 1, 'Edges Using K-Means', obj_var, obj_scale_func, 55, 0, 2, 0),
            self.Scale(tool_box, 0, 255, 1, 'Thresholding Value', thresh_var, thresh_scale_func, 33, 0, 1, 0)]

        '''____Empty_Labels____'''
        thresh_lbl = self.Label(tool_box, "binary", 33, 1, 1)
        self.Label(tool_box, "", 10, 0, 2)
        self.Label(tool_box, "-------------------------------------------------", 14, 0, 2)
        self.Label(tool_box, "", 18, 0, 2)
        self.Label(tool_box, "-------------------------------------------------", 29, 0, 2)
        self.Label(tool_box, "", 39, 0, 2)
        self.Label(tool_box, "-------------------------------------------------", 59, 0, 2)
        self.Label(tool_box, "-------------------------------------------------", 69, 0, 2)

        tool_box_canvas.create_window((0, 0), window=tool_box, anchor="nw")
        tool_box_canvas.configure(yscrollcommand=scrollbar.set)
        tool_box_canvas.bind('<Enter>', _bound_to_mousewheel)
        tool_box_canvas.bind('<Leave>', _unbound_to_mousewheel)
        tool_box.bind("<Configure>", lambda x: tool_box_canvas.configure(scrollregion=tool_box_canvas.bbox("all")))

        self.grid(row=0, column=0, padx=3, pady=5, sticky='nsew')
        tool_box_canvas.pack(side="left", fill="y")
        scrollbar.pack(side="left", fill="y")

        scales[2].update_state(state="disabled")

    class Button:
        def __init__(self, parent, text, func, param, row, column, cs, state="normal", img=""):
            # ima = PhotoImage(file=img).subsample(12)
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
        def __init__(self, parent, text, func, row, column):
            self.entry = Entry(parent)
            self.entry.insert(0, text)
            self.entry.bind('<Return>', func)
            self.entry.grid(row=row, column=column, padx=1, pady=1, sticky="nsew")

        def get(self):
            return self.entry.get()

        def reset(self):
            self.entry.delete(0, 'end')

        def set(self, text):
            self.entry.insert(0, text)

    class Label:
        def __init__(self, parent, text, row, column, cs):
            self.label = Label(parent, anchor='center')
            self.label.config(text=text)
            self.label.grid(row=row, column=column, columnspan=cs, sticky="nsew", padx=2, pady=1)

        def config(self, txt):
            self.label.config(text=txt)

        def get_text(self):
            return self.label.cget("text")

    class Scale:
        def __init__(self, parent, from_, to, resolution, label, variable, func, row, column, cs=1, init=0):
            self.scale = Scale(parent, orient='horizontal', cursor='hand2', from_=from_, to=to,
                               resolution=resolution,
                               label=label, variable=variable)  # , state="disabled", command=lambda x: func()
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


class ImageBox(Canvas):
    def __init__(self):
        super().__init__(width=int(screen_width * 0.85), height=int(screen_height * 0.95), borderwidth=0,
                         highlightthickness=0, bg='#C8C8C8')
        global image_box
        image_box = self
        self.grid(row=0, column=2, padx=0, pady=5, sticky='nsew')


class App(Tk):
    def __init__(self):
        super().__init__()
        global screen_width, screen_height
        screen_width, screen_height = int(self.winfo_screenwidth()), int(self.winfo_screenheight())

        def initialize_menu_bar():
            """____Menu_Functions____"""

            def menu_bar_rotate():
                angle = simpledialog.askinteger('Input', 'Enter Angle')
                if angle is None:
                    return
                rotate_with_angle(angle)

            def menu_bar_resize():
                width = simpledialog.askinteger('Input', 'Enter Width')
                height = simpledialog.askinteger('Input', 'Enter Height')
                if width is None or height is None:
                    return
                force_resize(width, height)

            menu_bar = Menu(self)
            file_menu = Menu(menu_bar, tearoff=0)
            settings_menu = Menu(menu_bar, tearoff=False)

            file_menu.add_command(label="Reset", command=lambda: open_image())
            file_menu.add_command(label="Credits", command=lambda: messagebox.showwarning('Credits',
                                                                                          "This App Was Created Under "
                                                                                          "The Supervision Of Dr. "
                                                                                          "Sabry AbdelMoaty By Ahmed "
                                                                                          "Aboelenin"))
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=lambda: self.exit())

            settings_menu.add_command(label="Another Rotate Angle", command=lambda: menu_bar_rotate())
            settings_menu.add_command(label="Force Resize", command=lambda: menu_bar_resize())
            settings_menu.add_command(label="Edit Filters Settings",
                                      command=lambda: messagebox.showwarning('Info', "Not Supported Yet! 🙃"))

            menu_bar.add_cascade(label="File", menu=file_menu)
            menu_bar.add_cascade(label="Settings", menu=settings_menu)
            return menu_bar

        '''____Main_Window___________________________________________________________________________________________'''
        self.title("Mi Image Toolbox")  # App Title
        self.iconphoto(False, PhotoImage(file='img/icon.png'))  # App Icon
        self.state('zoomed')  # Start The App Maximized
        self.geometry('+0+0')  # App Position
        self.minsize(int(screen_width * 0.6), int(screen_height))  # Minimum Size of The App Window
        self.maxsize(screen_width, screen_height)
        self.config(menu=initialize_menu_bar())

        '''____Initialize_ToolBox_And_ImageBox_______________________________________________________________________'''
        ToolBox()
        ImageBox()

    def start(self):
        open_image()
        self.mainloop()

    def exit(self):
        im.close_all_figures()
        sys.exit()

if __name__ == "__main__":
    App().start()
