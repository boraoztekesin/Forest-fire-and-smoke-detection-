from keras.layers import BatchNormalization
from keras.models import load_model
from keras.utils import img_to_array, load_img
import numpy as np

model_path = "my_model.h5"

# Loading the trained model
model = load_model(model_path, custom_objects={'BatchNormalization': BatchNormalization})

def predict_image(model, image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    prediction_label = 'smoke' if prediction > 0.5 else 'fire'

    return prediction_label

image_path = "C:/Users/Bora/Downloads/WildfireSmoke.jpg"
print(predict_image(model, image_path))
