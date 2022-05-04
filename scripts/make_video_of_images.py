# -*- coding: utf-8 -*-
"""

"""

import cv2
import os

if __name__ == "__main__":
    rev = True # to reverse the images, building instead of destructing
    folder_name = "s2000_copenhagen_relative_coverage"
    video_name = folder_name

    if rev is True:
        video_name = "../data/" + video_name + "_reverse.mp4"
    else:
        video_name = "../data/" + video_name + ".mp4"
    image_folder = "../data/" + folder_name
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    if rev is True:
        images.reverse() 
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 5, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()