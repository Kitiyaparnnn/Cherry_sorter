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
#from tkinter.ttk import *
from PIL import Image as Pil_image, ImageTk as Pil_imageTk

class FullScreenApp(object):
    def __init__(self, master, **kwargs):
        self.master = master
        pad = 3
        self._geom = '200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
        #master.bind('<Escape>', self.toggle_geom) 
          
    def toggle_geom(self, event):
        geom = self.master.winfo_geometry()
        self.master.geometry(self._geom)
        self._geom = geom
        
# --- Create main tkinter window ---
window = Tk()
app = FullScreenApp(window)
window.title("Cherry Sorter")
width = window.winfo_screenwidth()
height = window.winfo_screenheight()
print(width, height)

# --- Frame for main content ---
main_frame = Frame(window)
main_frame.pack(expand=True, fill="both")

# --- Adding image (PNG only) ---
image = Pil_image.open("/home/coffeecolor/Cherry_sorter/dataset/multiple_pi/image_1_20241214-130840.jpg")
resize_image = image.resize((int(width/2), int(720)))
img = Pil_imageTk.PhotoImage(resize_image)

# Label for the main image
image_label = Label(main_frame, image=img, justify = "center")
image_label.pack(side="left", expand=True, fill="both", )

# --- Adding logo image ---
logo_image = Pil_image.open("object_detection/cmu_logo.png")
resize_image = logo_image.resize((400, 400))
logo = Pil_imageTk.PhotoImage(resize_image)

# Frame for logo and text (right side)
right_frame = Frame(main_frame)
right_frame.pack(side="left", expand=True, fill="both", pady = 120)

logo_label = Label(right_frame, image=logo, justify="center")
logo_label.pack()  # Adds spacing around the logo

# --- Adding labels ---
l1 = Label(right_frame, text="Faculty of Engineering\nChiang Mai University", 
           justify="center", font=('Arial', 18, 'bold'))
l1.pack(pady=10)

l3 = Label(right_frame, text="Coffee Cherry Sorter", 
           justify="center", font=('Arial', 20, 'bold'))
l3.pack(pady=20)

l4 = Label(right_frame, text=f"Red cherries: 123, Green cherries: 234", 
           justify="center", font=('Arial', 16))
l4.pack(pady=10)

# --- Run the application loop ---
def exit(event):
       #stop_event.set()
       window.destroy()
    
window.bind("<Escape>", exit)
window.mainloop()
