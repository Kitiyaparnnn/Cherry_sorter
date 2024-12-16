import tkinter as tk
import cv2
from PIL import Image

#--- show image
# window = tk.Tk()
# window.geometry("400x300")

# frame = tk.Frame(window)
# frame.pack()

# label = tk.Label(frame, width=200, height=200)
# label.pack()

# cap = cv2.VideoCapture(0)

# def update_frame():
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.resize(frame, (200, 200)) # resize the frame to 200x200
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
#         imgtk = ImageTk.PhotoImage(image=img)
#         label.imgtk = imgtk
#         label.configure(image=imgtk)
#     window.after(10, update_frame)

# update_frame()
# window.mainloop()

# import tkinter module
from tkinter import * 
from tkinter.ttk import *
from PIL import Image as Pil_image, ImageTk as Pil_imageTk

# creating main tkinter window/toplevel
window = Tk()
window.title("Cherry Sorter")

# adding image (remember image should be PNG and not JPG)
img = PhotoImage(file = "test_img.png")
img1 = img.subsample(2, 2)
# setting image with the help of label
Label(window, image = img1).grid(row = 0, column = 0,
       columnspan = 2, rowspan = 5, padx = 5, pady = 5)

# adding logo image
logo_image = Pil_image.open("object_detection/cmu_logo.png")
resize_image = logo_image.resize((300, 300))
logo = Pil_imageTk.PhotoImage(resize_image)
Label(window, image = logo).grid(row = 0, column = 3,
       columnspan = 2, rowspan = 2, padx = 5, pady = 5)

# this will create a label widget
l1 = Label(window, text = "Faculty of Engineering\nChiang Mai University", justify="center",font=('Arial', 18,'bold'))
l2 = Label(window, text = "",justify='center',font=('Arial', 18,'bold'))
l3 = Label(window, text = "Coffee Cherry Sorter🍒",justify="center",font=('Arial', 20,'bold'))
l4 = Label(window, text = f"Red cherries: 123, Green cherries: 234",justify='center',font=('Arial', 16))
l1.grid(row = 2, column = 3,columnspan = 2)
l2.grid(row = 2, column = 3,columnspan = 2,sticky='N',pady=2)
l3.grid(row = 4, column = 3,columnspan = 2,sticky='N')
l4.grid(row = 4, column = 3,columnspan = 2,pady=2)

# infinite loop which can be terminated 
# by keyboard or mouse interrupt
mainloop()
