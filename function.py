import joblib
import face_recognition
import itertools
import os
import random
import cv2 as cv
import numpy as np
from datetime import date
from sklearn import svm

def colincsv(input_file, output_file, transform_row):
    from csv import reader
    from csv import writer
    import os
    with open(input_file, 'r') as robj, open (output_file, 'w', newline='') as wobj:
        csvreader = reader(robj)
        csvwriter = writer(wobj)
        for row in csvreader:
            transform_row(row, csvreader.line_num)
            csvwriter.writerow(row)
    os.remove(input_file)
    os.rename(output_file, input_file)


def attendance(img):

    if not "data" in os.listdir():
        os.chdir("../")
        os.chdir("../")
    model = joblib.load("knn")
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
        if name in person:
            pass
        else:
            person.append(*name)
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            image = cv.rectangle(image, (j[1], j[2]), (j[3], j[0]), color, 2)
            image = cv.putText(image, *name, (j[3], j[0]), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
    #print(person)
    cv.imwrite("attendance/"+date.today().strftime("%d.%m.%Y")+".jpg",image)
    att= [date.today().strftime("%d.%m.%Y")]
    for i in range(1, 83):
        r = "CO193"+str(i) if i>9 else "CO1930"+str(i)
        if r in person:
            att.append(1)
        else:
            att.append(0)
    print(att)
    colincsv("attendance.csv", "temp.csv", lambda row, line_num: row.append(att[line_num-1]))
    return True, no_of_faces
    #messagebox.showinfo("Result", str(len(person))+" students present")
    #return image

def train():
    if not "data" in os.listdir():
        os.chdir("../")
        os.chdir("../")
    train = "data/"
    encodings = []
    roll = []
    train_dir = os.listdir(train)
    for person in train_dir:
        if person.startswith("CO19"):
            pix = os.listdir(train + person)
            for person_img in pix:
                face = face_recognition.load_image_file(train + person + "/" + person_img)
                face_bounding_boxes = face_recognition.face_locations(face)
                if len(face_bounding_boxes) == 1:
                    face_enc = face_recognition.face_encodings(face)[0]
                    encodings.append(face_enc)
                    roll.append(person)
                else:
                    print(person + "\t\t" + person_img)
            print("done "+person)
    classifier = svm.SVC(gamma ='scale')
    classifier.fit(encodings, roll)
    joblib.dump(classifier, "newmodel")
    #messagebox.showinfo("Result", "Training Successful")
