import os
import face_recognition
from sklearn import svm
from sklearn.svm import SVC
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

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
#print(roll, encodings)
#classifier = DecisionTreeClassifier(criterion='gini',random_state=20)
classifier = KNeighborsClassifier(n_neighbors=5)
#classifier = svm.SVC(kernel='rbf')
#classifier = SVC(kernel='rbf',random_state=0)
#classifier = BernoulliNB()
classifier.fit(encodings, roll)
joblib.dump(classifier, "knn")