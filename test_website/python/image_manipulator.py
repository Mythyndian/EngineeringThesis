import os
import random
from tkinter import image_names
import cv2 as cv
import numpy as np


class ImageManipulator:
    def __init__(self, image_names: list) -> None:
        self._file_names = image_names
        self._images = []
        for file in self._file_names:
            self._images.append(cv.imread(os.path.join("images", "personal", file)))

    def rotate_images(self):
        counter = 0
        for image in self._images:
            rows, cols = image.shape[:2]
            rotation = random.randint(0, 360)
            rotation_matrix = cv.getRotationMatrix2D(
                ((cols - 1) / 2.0, (rows - 1) / 2.0), rotation, 1
            )
            rotated_img = cv.warpAffine(image, rotation_matrix, (cols, rows))
            cv.imwrite(
                os.path.join("images", "others", f"rotated_{rotation}_{counter}.jpg"),
                rotated_img,
            )
            counter += 1
            
        

    def transform_by_median_blur(self):
        counter = 0
        for image in self._images:
            image_blured = cv.medianBlur(image, 5)
            cv.imwrite(
                os.path.join("images", "others", f"median_blur_{counter}.jpg"),
                image_blured,
            )
            counter += 1


images = os.listdir(os.path.join("images", "personal"))
print(images)
im = ImageManipulator(images)
im.transform_by_median_blur()
im.rotate_images()
