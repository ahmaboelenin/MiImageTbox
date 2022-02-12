import sys
from tkinter import Menu, messagebox, simpledialog


def exit_app():
    sys.exit()


class MenuBar(Menu):
    def __init__(self, master=None):
        super().__init__(master=master)
        self.file_menu = Menu(self, tearoff=0)
        self.settings_menu = Menu(self, tearoff=0)
        self.help_menu = Menu(self, tearoff=0)

        self.file_menu.add_command(label="Reset", command=self.reset)
        self.file_menu.add_command(label="Another Rotate Angle", command=self.menu_bar_rotate)
        self.file_menu.add_command(label="Force Resize", command=self.menu_bar_resize)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.master.destroy)

        self.settings_menu.add_command(label="Edit Filters Settings",
                                       command=lambda: messagebox.showwarning('Info', "Not Supported Yet! 🙃"))

        self.help_menu.add_command(label="Credits",
                                   command=lambda: messagebox.showwarning("Credits", "This App was Created Under the "
                                                                                     "Supervision of Dr. Sabry "
                                                                                     "AbdelMoaty By Ahmed "
                                                                                     "Aboelenin"))

        self.add_cascade(label="File", menu=self.file_menu)
        self.add_cascade(label="Settings", menu=self.settings_menu)
        self.add_cascade(label="Help", menu=self.help_menu)

        self.master.configure(menu=self)

    def reset(self):
        self.master.imageHistory, self.master.index = [], 0
        self.master.toolBox.open('Sample')
        self.master.toolBox.clear_frame()
        self.master.toolBox.enable_elements()
        self.master.toolBox.reset_entries()

    def menu_bar_rotate(self):
        angle = simpledialog.askstring('Entry', 'Enter Angle')
        if angle is None:
            return
        try:
            angle = int(angle)
            self.master.toolBox.rotate(angle)
        except ValueError:
            return

    def menu_bar_resize(self):
        width = simpledialog.askinteger('Input', 'Enter Width')
        height = simpledialog.askinteger('Input', 'Enter Height')
        if width is None or height is None:
            return
        self.master.toolBox.force_resize(width, height)
