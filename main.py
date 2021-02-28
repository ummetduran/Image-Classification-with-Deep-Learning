import keras
from keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print(type(x_train))
print(x_train.shape)
datanum=1000
import matplotlib.pyplot as plt
#plt.imshow(x_train[datanum])

print("Label:",y_train[datanum])
from keras.utils import to_categorical

y_train_ohe=to_categorical(y_train)
y_test_ohe=to_categorical(y_test)

print(y_train_ohe)
x_train=x_train/255
x_test=x_test/255

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout

model=Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
    padding='same', input_shape=(32,32,3)))
model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
    padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
    padding='same'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
    padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
    padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
    padding='same'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#model.fit(x_train,y_train_ohe,batch_size=256,epochs=20,validation_split=0.3)
model = keras.models.load_model('model_4.h5')
#model.save('model_3.h5')


def İmageClassifiar(self):

  image=plt.imread(self)
  plt.imshow(image)

  from skimage.transform import resize
  reimage=resize(image,(32,32,3))
  plt.imshow(reimage)
  import numpy as np
  ols=model.predict(np.array([reimage,]))
  sınıflar=['Uçak','Otomobil','Kuş','Kedi','Geyik','Köpek','Kurbağa','At','Gemi','Kamyon']
  index=np.argsort(ols[0,:])

  f = plt.figure(figsize=(12, 6))

  for i in range(9,0,-1):
    print(round(ols[0,index[i]],3)," Olasılık ile sınıfı:",sınıflar[index[i]])
    plt.bar(sınıflar[index[i]],round(ols[0, index[i]],3))
  plt.show()

İmageClassifiar('ucak.jpg')

