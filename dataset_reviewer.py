import os
import shutil
from tkinter import Tk, Label
from PIL import Image, ImageTk

index_file = "dataset_viewer_index.txt"


image_dir = "patches"
bad_image_dir = "bad_patches"

os.makedirs(bad_image_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png'))]


if os.path.exists(index_file):
    with open(index_file, "r") as f:
        try:
            index = int(f.read().strip())
        except ValueError:
            index = 0
else:
    index = 0



root = Tk()
root.title("Dataset Reviewer")

label_text = Label(root, font=("Arial",16))
label_text.pack(pady=(10,5))

img_label = Label(root)
img_label.pack()

def extract_label(filename):
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[1].rsplit(".",1)[0]
    return "Unknown"

def save_index():
    with open(index_file, "w") as f:
        f.write(str(index))

def show_image():
    if index >= len(image_files):
        print("No more patches")
        root.quit()
        return
    
    filename = image_files[index]
    img_path = os.path.join(image_dir, filename)
    
    try:
        label = extract_label(filename)
        label_text.config(text=f"Label: {label} ID: {filename}")
        
        img = Image.open(img_path)
        img.thumbnail((800,800))
        tk_img = ImageTk.PhotoImage(img)
        img_label.config(image=tk_img)
        img_label.image = tk_img
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        skip_image()

def mark_bad(event=None):
    global index
    src = os.path.join(image_dir, image_files[index])
    dst = os.path.join(bad_image_dir, image_files[index])
    shutil.move(src,dst)
    index += 1
    save_index()
    show_image()

def skip_image(event=None):
    global index
    index += 1
    save_index()
    show_image()
    

root.bind("<Left>",mark_bad)
root.bind("<Right>",skip_image)
root.bind("<space>",skip_image)

show_image()
root.mainloop()

