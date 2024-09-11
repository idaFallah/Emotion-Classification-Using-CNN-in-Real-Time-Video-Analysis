
# import libs

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab.patches import cv2_imshow
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization

from google.colab import drive
drive.mount('/content/drive')

path = '/content/drive/MyDrive/Computer Vision/Datasets/fer_images.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

tf.keras.preprocessing.image.load_img('/content/fer2013/train/Disgust/104.jpg')

image = tf.keras.preprocessing.image.load_img('/content/fer2013/train/Neutral/1027.jpg')
image

# train/ test generator

training_generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range=7,
                                        horizontal_flip=True,
                                        zoom_range=0.2)
train_dataset = training_generator.flow_from_directory('/content/fer2013/train',
                                                        target_size = (48, 48),
                                                        batch_size = 16,
                                                        class_mode = 'categorical',
                                                        shuffle = True)

train_dataset.classes

np.unique(train_dataset.classes, return_counts=True)  # the amount of pics for each of the classes

train_dataset.class_indices

sns.countplot(x = train_dataset.classes);

test_generator = ImageDataGenerator(rescale=1./255)
test_dataset = test_generator.flow_from_directory('/content/fer2013/validation',
                                                  target_size = (48, 48),
                                                  batch_size = 1,
                                                  class_mode = 'categorical',
                                                  shuffle = False)

# building & training the CNN

num_detectors = 32
num_classes = 7
width, height = 48, 48
epochs = 100

network = Sequential()

network.add(Conv2D(num_detectors, (3, 3), activation='relu', padding='same', input_shape=(width, height, 3)))
network.add(BatchNormalization())
network.add(Conv2D(num_detectors, (3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*num_detectors, (3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*num_detectors, (3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*num_detectors, (3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*num_detectors, (3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*2*num_detectors, (3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*2*num_detectors, (3, 3), activation='relu', padding='same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Flatten())

# hidden layers:

network.add(Dense(2*num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Dense(num_classes, activation='softmax'))
print(network.summary())

network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

network.fit(train_dataset, epochs=epochs)  # takes 1 hour to run

# saving & loading the model

model_json = network.to_json()
with open('network_emotions.json','w') as json_file:
  json_file.write(model_json)

from keras.models import save_model
network_saved = save_model(network, '/content/weights_emotions.hdf5')

with open('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Weights/network_emotions.json', 'r') as json_file:
  json_saved_model = json_file.read()
json_saved_model

network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Weights/weights_emotions.hdf5')
network_loaded.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

network_loaded.summary()

# evaluating the NN

network_loaded.evaluate(test_dataset)

predictions = network_loaded.predict(test_dataset)
predictions

predictions = np.argmax(predictions, axis=1)
predictions

test_dataset.classes

from sklearn.metrics import accuracy_score
accuracy_score(test_dataset.classes, predictions)  # result is the same as "network_loaded.evaluate(test_dataset)"

test_dataset.class_indices

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_dataset.classes, predictions)
cm

sns.heatmap(cm, annot=True)

from sklearn.metrics import classification_report
print(classification_report(test_dataset.classes, predictions))

# classifying one single image

image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/gabriel.png')
 cv2_imshow(image)

image.shape

# since our NN is trained on only images of faces and not extra stuff, we need to detect faces first the detect emotions

face_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Computer Vision/Cascades/haarcascade_frontalface_default.xml')

original_image = image.copy()
faces = face_detector.detetcMultiScale(original_image)

faces

# extracting only the face part of the image

roi = image[40:40 + 128, 162:162 + 128]              # Region Of Interest -> x of face + width , y of face + height
cv2_imshow(roi)

roi.shape

# resizing : our model was trained using 48 * 48 images

roi = cv2.resize(roi, (48, 48))
cv2_imshow(roi)

roi.shape

# normalizing image values

roi = roi / 255
roi

roi.shape

# adding a dimension to the image to send it to NN (batch format)

roi = np.expand_dims(roi, axis=0)  # axis=0 -> where the dimesnion is added
roi.shape

probs = network_loaded.predict(roi)
probs

result = np.argmax(probs)
result

test_dataset.class_indices

# classifying multiple images

image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/faces_emotions.png')
 cv2_imshow(image)

faces = face_detector.detectMultiScale(image)
faces

test_dataset.class_indices.keys()   # keys() to get only the name of classes

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# drawing a binding box around faces

for(x, y, w, h) in faces:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  roi = image[y:y + h, x:x + w]
  #cv2_imshow(roi)
  roi = cv2.resize(roi, (48, 48))
  cv2_imshow(roi)
  roi = roi / 255
  roi = np.expand_dims(roi, axis=0)
  #print(roi.shape)
  prediction = network_loaded.predict(roi)
  #print(prediction)
  cv2.putText(image, emotions[np.argmax(prediction)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
cv2_imshow(image)

# classifying emotions in videos

cap = cv2.VideoCapture('/content/drive/MyDrive/Computer Vision/Videos/emotion_test01.mp4')
connected, video = cap.read()
print(connected, video.shape)

save_path = '/content/drive/MyDrive/Computer Vision/Videos/emotion_test01_myResult.avi'
# saving directly to the google drive to not lose to path

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # defining the codec to save the video
fps = 24
output_video = cv2.VideoWriter(save_path, fourcc, fps, (video.shape[1], video.shape[0]))

while (cv2.waitKey(1) < 0):
  connected, frame = cap.read()
  if not connected:
    break
  faces= face_detector.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))
  if len(faces) > 0:
    for(x, y, w, h) in faces:
      frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      roi = frame[y:y + h, x:x + w]
      roi = cv2.resize(roi, (48, 48))
      cv2_imshow(roi)
      roi = roi / 255
      roi = np.expand_dims(roi, axis=0)
      prediction = network_loaded.predict(roi)

      if prediction is not None:
        result = np.argmax(prediction)
        cv2.putText(frame, emotions[result], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

  cv2_imshow(frame)
  output_video.write(frame)

print('End')
output_video.release()   # to release the memory
cv2.destroyAllWindows()











