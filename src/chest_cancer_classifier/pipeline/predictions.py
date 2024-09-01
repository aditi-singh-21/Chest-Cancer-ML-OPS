import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredicationPipeline:
    def __init__(self,filename):
        self.filename = filename
        
        
    def predict(self):
        model = load_model(r"C:\Users\aditi\OneDrive\Desktop\Chest-Cancer-ML-OPS\artifacts\training\model.h5")
        image_name = self.filename
        test_image = image.load_img(image_name,target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        result = np.argmax(model.predict(test_image),axis=1)
        print(result)
        
        if result[0] == 0:
            prediction = "Adenocarcinoma Cancer"
        elif result[0] == 1:
            prediction = "Normal"
        elif result[0] == 2:
            prediction = "Squamous Cell Carcinoma"
        elif result[0] == 3:
            prediction = "Large Cell Carcinoma"
        else:
            prediction = "Unknown"

        return [{"image": prediction}]
            
        