import glob
import os
import re
from tkinter.filedialog import askdirectory, asksaveasfilename

import cv2
from tqdm import tqdm


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


image_folder = askdirectory(title="Open folder with images")
video_name = asksaveasfilename(
    defaultextension=".avi",
    filetypes=[("AVI", "*.avi")],
    title="Save video",
    initialfile="animation.avi",
)


images = natural_sort([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(filename=video_name, fourcc=0, fps=1, frameSize=(width, height))

for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
