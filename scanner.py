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


def createParser():
    parser = argparse.ArgumentParser()
#    parser.add_argument('--start', action='store_true')
    parser.add_argument('--image', nargs='?')
    parser.add_argument('--templates_file', nargs='?')
    parser.add_argument('--templates_dir', nargs='?')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--dir', nargs='?')
    parser.add_argument('--not_save', action='store_true')
    return parser


def gray_convert(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

def blur(img):
    return cv2.medianBlur(img, 5)


def erode(img):
    return cv2.erode(img, np.ones((3, 3), np.uint8), iterations=1)


def turn(img, angle):
    return ndimage.rotate(img, angle)


def bil_filter(img):
    return cv2.bilateralFilter(img, -1, 13, 13)


def text_to_words(txt):
    text = re.sub('[\n\t\r,<>;:+?"/\\^%#@№~`*&!=|_©\[\]\{\}\(\)]', ' ', txt)
    words = re.split(r'\s+', text.lower())
    return set(words)


def img_to_text(img, cfg):
    return pytesseract.image_to_string(
                    img,
                    lang='rus',
                    config=cfg
    )

def img_to_letters(img):
    letters = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        if r > 0.4 and w > 8 and h > 8:
            if w > 500 and h > 500:
                def_img = img[y:y+h-1, x:x+w-1]
                letters += img_to_letters(def_img)
                letters.append(def_img)
            elif w > h*2:
                def_img = img[y:y+h-1, x:x+w-1]
            elif h > w*2:
                def_img = turn(img[y:y+h-1, x:x+w-1], -90)
                letters.append(def_img)
    return letters


class Definer():

    def __init__(self, file, training=0, not_save=0):
        self.training = training
        self.not_save = not_save
        if isinstance(file, list):
            self.path = file[len(file)-1]
            self.knowlage = dict()
            for elem in file[:-1]:
                with open(self.path + '/' + elem, 'r') as f:
                    self.knowlage.update({elem[:-4]: yaml.load(f)})
        elif isinstance(file, str):
            self.file = file
            with open(self.file, 'r') as f:
                self.knowlage = yaml.load(f)

    def doc_def(self, words, file_name):
        for doc in self.knowlage:
            size = 0
            weight = 0
            train_w = []
            for word in self.knowlage[doc]:
                if word != "koef":
                    size += self.knowlage[doc][word]
                    if word in words:
                        train_w += [word]
                        weight += self.knowlage[doc][word]
                else:
                    koef = self.knowlage[doc][word]
            if float(koef) <= weight/size:
                if self.training:
                    self.train(doc, train_w, file_name)
                return doc

    def train(self, doc, words, file_name):
        p = 1 if re.match(doc, file_name) else -1
        for word in words:
            self.knowlage[doc].update({word: self.knowlage[doc][word] + p})
        self.save_base(doc)

    def save_base(self, name):
        if self.not_save:
            print(yaml.dump(self.knowlage, allow_unicode=True))
        else:
            try:
                if self.file:
                    with open(self.file, 'w') as f:
                        yaml.dump(self.knowlage, f, allow_unicode=True)
            except AttributeError:
                with open(self.path+'/'+name+'.yml', 'w') as f:
                    yaml.dump(self.knowlage[name], f, allow_unicode=True)

    def define_image(self, filename):
        time_start = time.time()
        pages = pdf2image.convert_from_path(
                            filename,
                            thread_count=2,
                            fmt="png",
                            dpi=300
        )
        time_stop = time.time()
#       print(time_stop - time_start)
        i = 0
        for page in pages:
            i += 1
            print("Page ", i)
            image = np.array(page)
            image = gray_convert(image)
            image = bil_filter(image)
            res, words = self.define_page(image, filename.split('/')[-1])
            if not (res is None):
                print(res)
            else:
                for letter in img_to_letters(image):
                    words.update(text_to_words(img_to_text(letter, '--psm 6 --oem 1')))
                print(words)
                print(self.doc_def(words, filename.split('/')[-1]))

    def define_page(self, img, filename):
        time_start = time.time()
        text = img_to_text(img, '')
        time_stop = time.time()
#          print(time_stop - time_start)
        words = text_to_words(text)
        return self.doc_def(words, filename), words


def main(namespace):
    if namespace.train:
        train = 1
    else:
        train = 0
    if namespace.not_save:
        not_save = 1
    else:
        not_save = 0
    if namespace.templates_file:
        definer = Definer(namespace.templates_file, train, not_save)
    elif namespace.templates_dir:
        definer = Definer(
                os.listdir(path=namespace.templates_dir) +
                [namespace.templates_dir],
                train,
                not_save
        )
    else:
        definer = Definer('base.yml', train, not_save)
    if namespace.image:
        definer.define_image(namespace.image)
    elif namespace.dir:
        for filename in os.listdir(path=namespace.dir):
            print("File", filename)
            definer.define_image(namespace.dir + '/' + filename)


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args()
    main(namespace)
