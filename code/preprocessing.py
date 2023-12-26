import cvlib as cv
import cv2
import numpy as np
import os

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def Cutting_face_save(image, name):
    faces, conf = cv.detect_face(image)
    print(faces, conf, len(faces))
    for (x, y, x2, y2) in faces:
        cropped = image[y:y2, x:x2]
        resize = cv2.resize(cropped, (96,96))
        img = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        imwrite(name+".jpg", img)

path_dir = 'no\\'
file_list = os.listdir(path_dir)

file_name_list = []
for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg", ""))

for name in file_name_list:
    image = imread("no\\"+name+".jpg")
    Cutting_face_save(image, name)



