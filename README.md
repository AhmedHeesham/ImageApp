# ImageApp
image processing 

# Creating a GUI that allows the user to:
- Upload an image
- Calculate its histogram and display it
- Apply histogram equalization and display both equalized image and its histogram
- Apply filtering (Sobel, Laplace) + user types parameters and display them
- Apply Fourier Transform of image and display it
- Add noise (Salt and pepper, Periodic) + user types parameters and display noisy image
- Remove S&P using median + user types parameters and display clean image
- Remove periodic noise (user selects method: Notch/Band-reject/Mask), for mask method user is allowed to select 2 pixels on Fourier Transform of noisy image display for you to remove it.
- Notch/band reject: you detect and remove noise AUTOMATICALLY. User will NOT give any coordinates. Then you remove noise.
- Mask: user only SELECTS 2 pixels. User will NOT give any coordinates. Just SELECTS with a MOUSE CLICK.Then you remove noise.
