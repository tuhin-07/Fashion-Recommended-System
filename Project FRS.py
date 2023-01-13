import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm

model = ResNet50(weights="imagenet",include_top=False , input_shape=(224,224,3))
model.trainable = False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
    ])

print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_arr= image.img_to_array(img)
    expanded_img_arr = np.expand_dims(img_arr,axis=0)
    preprocessed_img= preprocess_input(expanded_img_arr)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    
    return normalized_result

filename=[]
for file in os.listdir(r"C:\Users\Tuhin\Downloads\images 2"):
    filename.append(os.path.join(r"C:\Users\Tuhin\Downloads\images 2",file))
    
feature_list=[]
for file in tqdm(filename):
    feature_list.append(extract_features(file, model))
    
# print(np.array(feature_list).shape)

pickle.dump(feature_list,open("embeddings.pkl",'wb'))
pickle.dump(filename,open("filename.pkl",'wb'))
