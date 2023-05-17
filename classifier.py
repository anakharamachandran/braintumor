import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2

def read_image(imgname):

    image = np.asarray(Image.open(imgname).resize((224,224)))
    return image

def predict_x(x):
    
    model = keras.models.load_model('model/xceptionpre.h5')
    p = model.predict(x)
    return p

def classify(imgname):

   
    # img = tf.keras.utils.load_img(
    # "static/uploads/"+imgname, target_size=(224,224)
    # )
    # img_array = tf.keras.utils.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0) # Create a batch
    model = keras.models.load_model('model/xceptionpre.h5')
    # predictions = model.predict(img_array)
    # score = tf.nn.softmax(predictions[0])
    # print(np.argmax(score))
    # return np.argmax(score)
    # Load the test image
    img = cv2.imread("static/uploads/"+imgname)

    # Preprocess the image
    img = cv2.resize(img, (224, 224))# resize to match model input size
    pre_img=img 
    cv2.imwrite("static/uploads/image.jpg", pre_img)
    print("image shape",pre_img.shape)

    img = np.expand_dims(img, axis=0) # add batch dimension
    img = img / 255.0 # normalize pixel values
    predictions = model.predict(img)
    score = tf.nn.softmax(predictions[0])
    return np.argmax(score),pre_img

   
