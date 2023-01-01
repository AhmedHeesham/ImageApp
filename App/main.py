import tkinter
import customtkinter
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from skimage import img_as_ubyte, img_as_float
from skimage import data, io, filters
from matplotlib.pyplot import imshow, show, subplot, title, get_cmap, hist
from scipy import ndimage
from scipy.fftpack import fft, fft2, fftshift, ifftshift, ifft2

from skimage.util import random_noise

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):

    def __init__(self):
        super().__init__()
        # configure window
        self.title("Image Project")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create left frame
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Image Project",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Upload",
                                                        command=self.Upload_image)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="histogram equalization",
                                                        command=self.sidebar_button_event)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Sobel and Laplace",
                                                        command=self.sidebar_button_event)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                               values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))


        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=3, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Histogram")
        self.tabview.add("periodic noise")
        self.tabview.add("salt&paper")

        self.tabview.tab("Histogram").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("periodic noise").grid_columnconfigure(0, weight=1)
        self.tabview.tab("salt&paper").grid_columnconfigure(0, weight=1)


        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Histogram"), text="Histogram calculate",
                                                           command=self.Calculate_histogram)
        self.string_input_button.grid(row=0, column=0, padx=20, pady=(10, 10))

        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Histogram"), text="Histogram Equalization",
                                                           command=self.Histogram_Equalizatioon)
        self.string_input_button.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Histogram"), text="Plot",
                                                           command=self.Histogram_Equalizatioon)
        self.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("periodic noise"),
                                                   text="Notch/Band-reject",
                                                   command=self.Remove_periodic_noise_Notch)
        self.label_tab_2.grid(row=0, column=0, padx=20, pady=(10, 10))

        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("periodic noise"), text="Mask",
                                                   command=self.Remove_periodic_noise_Mask)
        self.label_tab_2.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("periodic noise"), text="Plot",
                                                   command=self.Remove_periodic_noise_Mask)
        self.label_tab_2.grid(row=2, column=0, padx=20, pady=(10, 10))

        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("salt&paper"), text="Add",
                                                   command=self.add_salt_paper_noise)
        self.label_tab_2.grid(row=0, column=0, padx=20, pady=(10, 10))

        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("salt&paper"), text="Remove",
                                                   command=self.add_salt_paper_noise)
        self.label_tab_2.grid(row=1, column=0, padx=20, pady=(10, 10))

        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("salt&paper"), text="Plot",
                                                   command=self.add_salt_paper_noise)
        self.label_tab_2.grid(row=2, column=0, padx=20, pady=(10, 10))

        # create right frame

        # self.radiobutton_frame = customtkinter.CTkFrame(self)
        # self.radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # set default values

        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

        self.textbox.insert("0.0", "Creating a GUI that allows the user to:\n"
                                   "- Upload an image\n- Calculate its histogram and display it\n"
                                   "- Apply histogram equalization and display both equalized image and its histogram\n"
                                   "- Apply filtering (Sobel, Laplace) + user types parameters and display them\n"
                                   "- Apply Fourier Transform of image and display it\n"
                                   "- Add noise (Salt and pepper, Periodic) + user types parameters and display noisy image\n"
                                   "- Remove S&P using median + user types parameters and display clean image\n"
                                   "- Remove periodic noise (user selects method: Notch/Band-reject/Mask), for mask method user is allowed to select 2 pixels on Fourier Transform of noisy image display for you to remove it.\n"
                                   "- Notch/band reject: you detect and remove noise AUTOMATICALLY. User will NOT give any coordinates. Then you remove noise.\n"
                                   "- Mask: user only SELECTS 2 pixels. User will NOT give any coordinates. Just SELECTS with a MOUSE CLICK.Then you remove noise.")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def Calculate_histogram(self):
        # Calculate the histogram
        histogram = np.histogram(image, bins=np.arange(0, 255))

        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot the data
        ax.set_title('Histogram of grey values')
        ax.plot(histogram[1][:-1], histogram[0])

        # Show the plot
        plt.show()

        return histogram

    def sidebar_button_event(self):
        print("sidebar_button click")

    def Upload_image(self):
        file_path = tkinter.filedialog.askopenfilename()
        global image,image2
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = image2
        return image

    def Remove_periodic_noise_Mask(self):
        print("")

    def Remove_periodic_noise_Notch(self):
        print("")

    def add_periodic_noise(self):
        shape = image.shape[0], image.shape[1]
        noise = np.zeros(shape, dtype='float64')

        x, y = np.meshgrid(range(0, shape[0]), range(0, shape[1]))
        s = 1 + np.sin(x + y / 1.5)
        noisy_periodic_Image = ((image) / 128 + s) / 4

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(16, 8)
        ax1.imshow(image, 'gray')
        ax2.imshow(noisy_periodic_Image, 'gray')

        plt.show()

    def add_salt_paper_noise(self):
        global noise_Salt_Paper_img
        noise_Salt_Paper_img = random_noise(image, mode='s&p', amount=0.3)

        noise_Salt_Paper_img = np.array(255 * noise_Salt_Paper_img, dtype='uint8')

        # Display the noise image
        plt.imshow(noise_Salt_Paper_img, 'gray')
        cv2.waitKey(0)

    def remove_salt_and_pepper(self):
        global ImgNewwithoutnoise
        ImgNewwithoutnoise = np.median(noise_Salt_Paper_img, disk(3))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(16, 8)
        ax1.imshow(noise_Salt_Paper_img, 'gray')
        ax2.imshow(ImgNewwithoutnoise, 'gray')
        plt.show()

    def Histogram_Equalizatioon(self):
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()

        plt.plot(cdf_normalized, color='b')
        plt.hist(image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    def Plot_Fun(self):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
        ax1.imshow(image2, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax2.imshow(image, cmap=plt.cm.gray)


if __name__ == "__main__":
    app = App()
    app.mainloop()
