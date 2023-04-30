#####################
# Importing Libraries
#####################

import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime
import pyttsx3 as t2s

engine = t2s.init()

#####################
# Functions
#####################

def resize(img, size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimensions = (width, height)
    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)


def find_encoding(images):
    encoding_list = []
    for img in images:
        img = resize(img, 0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode_image = face_rec.face_encodings(img)[0]
        encoding_list.append(encode_image)
    return encoding_list


def present(name):
    with open('attendance.csv', 'r+') as file:
        data_list = file.readlines()
        name_list = []
        for i in data_list:
            entry = i.split(',')
            # 0 will give names and 1 will give time from csv file
            name_list.append(entry[0])

        # if name exists in csv, then it won't add that again
        if name not in name_list :
            time = datetime.now()
            time_string = time.strftime('%H: %M')
            file.writelines(f'\n{name}, {time_string}')
            statement = str('Welcome '+name)
            engine.say(statement)
            engine.runAndWait()

#####################
# Variables
#####################
path = 'students'
student_images = []
student_name = []
my_list = os.listdir(path)
# print(my_list)


#####################
# Program
#####################

for i in my_list :
    now = cv2.imread(f'{path}\{i}')
    student_images.append(now)
    student_name.append(os.path.splitext(i)[0])
# print(student_name)

encode_list = find_encoding(student_images)

vid = cv2.VideoCapture(0)

while True :
    success, frame = vid.read()
    Smaller_frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)

    # code for multiple faces in frame
    faces_in_frame = face_rec.face_locations(Smaller_frames)
    encode_faces_in_frame = face_rec.face_encodings(Smaller_frames,faces_in_frame)

    for encode_face, face_location in zip(encode_faces_in_frame, faces_in_frame):
        matches = face_rec.compare_faces(encode_list, encode_face)
        distance = face_rec.face_distance(encode_list, encode_face)
        print(distance)
        match_index = np.argmin(distance)

        # if a face matches then a rectangle is made on that face
        if matches[match_index] :
            name = student_name[match_index].upper()
            y1, x2, y2, x1 = face_location
            # all these points are in smaller frames (0.25 of original frame), but we want to show original frame, so points*4
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0),3)
            cv2.rectangle(frame, (x1, y2-30), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255,255), 2)
            present(name)

    cv2.imshow('CAPTURE',frame)
    cv2.waitKey(1)