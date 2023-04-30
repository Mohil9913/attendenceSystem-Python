#####################
# Importing Libraries
#####################

import cv2
import numpy as np
import face_recognition as face_rec


#####################
# Functions
#####################

def resize(img, size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimensions = (width, height)
    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)


#####################
# Image Declaration
#####################

# Loading image
mohil1 = face_rec.load_image_file('profile\mohil1.jpg')
mohil2 = face_rec.load_image_file('profile\mohil2.jpg')

# Converting to RGB
mohil1 = cv2.cvtColor(mohil1, cv2.COLOR_BGR2RGB)
mohil2 = cv2.cvtColor(mohil2, cv2.COLOR_BGR2RGB)

# resizing image
mohil1 = resize(mohil1, 0.5)
mohil2 = resize(mohil2, 0.5)


#####################
# Finding Face from Image
#####################

# Finding location
face_location_mohil1 = face_rec.face_locations(mohil1)[0]
face_location_mohil2 = face_rec.face_locations(mohil2)[0]

# Encoding face measurements
encode_mohil1 = face_rec.face_encodings(mohil1)[0]
encode_mohil2 = face_rec.face_encodings(mohil2)[0]

# Put Rectangle on Face (img, rectangle_measures, frame_color, frame_width)
cv2.rectangle(mohil1, (face_location_mohil1[3], face_location_mohil1[0]), (face_location_mohil1[1], face_location_mohil1[2]), (255, 255,0), 3)
cv2.rectangle(mohil2, (face_location_mohil2[3], face_location_mohil2[0]), (face_location_mohil2[1], face_location_mohil2[2]), (255, 255,0), 3)


result = face_rec.compare_faces([encode_mohil1], encode_mohil2)
print(result)
cv2.putText(mohil2, f'{result}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Show images and wait until closed
cv2.imshow('front_img', mohil1)
cv2.imshow('side_img', mohil2)
cv2.waitKey(0)
cv2.destroyAllWindows()

