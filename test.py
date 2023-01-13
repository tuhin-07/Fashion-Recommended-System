import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
#import cv2

feature_list= np.array(pickle.load(open('embeddings.pkl','rb')))

# print(feature_list)

filenames= pickle.load(open('filename.pkl','rb'))

model = ResNet50(weights="imagenet",include_top=False , input_shape=(224,224,3))
model.trainable = False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
    ]) 

img = image.load_img(r"C:\Users\Tuhin\Downloads\sample\1163.jpg",target_size=(224,224))
img_arr= image.img_to_array(img)
expanded_img_arr = np.expand_dims(img_arr,axis=0)
preprocessed_img= preprocess_input(expanded_img_arr)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors= NearestNeighbors(n_neighbors=6,algorithm='brute' , metric='euclidean')
neighbors.fit(feature_list)

distance , indices = neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0][1:6]:
    print(filenames[file])
#    temp_img=cv2.imread(filename[file])
#    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
#    cv2.waitkey(0)
    