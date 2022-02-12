from tkinter import Tk, PhotoImage
from menuBar import MenuBar
from toolBox import ToolBox
from imageBox import ImageBox


class App(Tk):
    def __init__(self):
        super().__init__()
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        self.image, self.ratio = None, 0
        self.imageHistory, self.index = [], 0

        self.register = True
        self.undo = False
        self.crop = False

        # App Title & Icon
        self.title("Mi Image Toolbox")
        self.iconphoto(False, PhotoImage(file='img/icon.png'))

        # Start The App Maximized
        self.state('zoomed')

        # App Position
        self.geometry('+0+0')

        self.minsize(int(self.screen_width * 0.5), int(self.screen_height))
        self.maxsize(self.screen_width, self.screen_height)

        MenuBar(self)
        self.imageBox = ImageBox(self)
        self.toolBox = ToolBox(self)

        self.toolBox.grid(row=0, column=0, padx=0, pady=0, sticky='nsew')
        self.imageBox.grid(row=0, column=1, padx=0, pady=0, sticky='nsew')

    def start(self):
        self.toolBox.open('Sample')
        if self.image is None:
            self.toolBox.init()
        self.mainloop()


if __name__ == "__main__":
    App().start()
