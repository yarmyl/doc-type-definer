#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pdf2image
import numpy as np
import pytesseract
import argparse
import cv2
import time
import re
import yaml
import os
from scipy import ndimage
import random


def createParser():
    parser = argparse.ArgumentParser()
#    parser.add_argument('--start', action='store_true')
    parser.add_argument('--image', nargs='?')
    parser.add_argument('--pdf', nargs='?')
    return parser


def gray_convert(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

def blur(img):
    return cv2.medianBlur(img, 3)

def erode(img):
    return cv2.erode(img, np.ones((3, 3), np.uint8), iterations=1)


def turn(img, angle):
    return ndimage.rotate(img, angle)
    
def filter(img):
    return cv2.bilateralFilter(img, -1, 13, 13)


def img_to_letters(img, save_img):
    letters = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    i = 0
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        if r > 0.4 and w > 8 and h > 8:
            i += 1
            if w > 500 and h > 500:
                def_img = img[y:y+h-1, x:x+w-1]
                letter, save_img = img_to_letters(def_img, save_img)
                letters += letter
            elif w > h*2:
                def_img = img[y:y+h-1, x:x+w-1]
                def_img = cv2.pyrUp(def_img)
                letters.append(def_img)
                cv2.rectangle(save_img, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
#                cv2.imwrite('temp/'+str(random.randint(0,1000000000))+'_test.png', def_img)
            elif h > w*2:
                def_img = turn(img[y:y+h-1, x:x+w-1], -90)
                def_img = cv2.pyrUp(def_img)
                letters.append(def_img)
                cv2.rectangle(save_img, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
#                cv2.imwrite('temp/'+str(random.randint(0,1000000000))+'_test.png', def_img)
    return (letters, save_img)

def img_to_text(img):
    image = gray_convert(img)
    image = filter(image)
    letters, save_img = img_to_letters(image, image.copy())
    cv2.imwrite('test.png', save_img)
    for letter in letters:
        cv2.imwrite('temp/'+str(random.randint(0,1000000000))+'_test.png', letter)
        print(pytesseract.image_to_string(
                    letter,
                    lang='rus',
                    config='--psm 6 --oem 1',
        ))
    return pytesseract.image_to_string(
                    letter,
                    lang='rus',
                    config='--psm 6 --oem 1',
    )


def to_image(filename):
    pages = pdf2image.convert_from_path(
                        filename,
                        thread_count=2,
                        fmt="png",
                        dpi=300,
#                        grayscale=True
    )
    return pages


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args()
    start = time.time()
    if namespace.pdf:
        images = to_image(namespace.pdf)
    if namespace.image:
        images = [cv2.imread(namespace.image)]
    for image in images:
        print("parse image")
        """
        image = cv2.pyrUp(image)
        print(pytesseract.image_to_string(
                    image,
                    lang='rus',
                    config='--psm 6 --oem 1',
        ))
        """
        print("Page", img_to_text(np.array(image)))
    stop = time.time()
    print(stop - start)
#    file = open('text.pdf', 'wb')
#    for line in open('text.txt', 'rb').readlines():
#        file.write(line)
#    file.close()
