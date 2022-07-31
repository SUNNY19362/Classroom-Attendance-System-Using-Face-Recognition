import joblib
import face_recognition
import cv2 as cv
import numpy as np
import os
import itertools
import random
from datetime import date
import sys

model = joblib.load(sys.argv[1])
img='testimages/c0.jpg'
test_image = face_recognition.load_image_file(img)
face_locations = face_recognition.face_locations(test_image)
no_of_faces = len(face_locations)
person = []
#file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
#image = cv.imdecode(file_bytes, 1)
image = cv.imread(str(img))
print(no_of_faces, face_locations)
for i,j in zip(range(no_of_faces), face_locations):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = model.predict([test_image_enc])
    print(name)
    person.append(*name)
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    image = cv.rectangle(image, (j[1], j[2]), (j[3], j[0]), color, 2)
    image = cv.putText(image, *name, (j[3], j[0]), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
#print(person)
cv.imwrite("attendance/"+str(sys.argv[1])+date.today().strftime("%d.%m.%Y")+".jpg",image)
att= [date.today().strftime("%d%m%Y")]
for i in range(1, 83):
    r = "CO193"+str(i) if i>9 else "CO1930"+str(i)
    if r in person:
        att.append(1)
    else:
        att.append(0)
print(att)