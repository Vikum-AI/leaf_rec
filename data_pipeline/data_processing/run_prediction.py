from keras.models import load_model
from cv2 import imdecode, IMREAD_COLOR 
import numpy as np
from keras.utils import img_to_array
import tensorflow as tf

def prediction(file):
    model = load_model('./models')
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = imdecode(file_bytes, IMREAD_COLOR)

    resize = tf.image.resize(img, (256,256))

    yhat = model.predict(np.expand_dims(resize/255, 0))
    class_names = ['Balloon vine', 'Coriander', 'Karanda', 'Lemon', 'Mint', 'Mustard', 'Not a matching herb','Oleander', 'Pomegranate']
    
    # probabilities = model.predict(resize)
    image_array = img_to_array(resize)
    image = np.reshape(image_array,[1,256,256,3])
    
    # Get the predicted output probabilities for the input image
    output_probabilities = model.predict(image)
    # Get the index of the maximum value in the output array
    predicted_class_index = np.argmax(output_probabilities)
    # Get the name of the predicted class
    predicted_class_name = class_names[predicted_class_index]
    # Print the predicted class name
    return predicted_class_name
