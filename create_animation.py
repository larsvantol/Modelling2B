import glob
import os
from tkinter.filedialog import askdirectory, asksaveasfilename

import cv2
from tqdm import tqdm

# TODO
image_folder = askdirectory(title="Open folder with images")
video_name = asksaveasfilename(
    defaultextension=".avi",
    filetypes=[("AVI", "*.avi")],
    title="Save video",
    initialfile="animation.avi",
)


images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(filename=video_name, fourcc=0, fps=1, frameSize=(width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

# Open the video in VLC
os.system(f"vlc {video_name}")
