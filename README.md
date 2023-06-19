# Forest Fire Detection
Forest fire detection
This project trains a Convolutional Neural Network (CNN) model using Keras and TensorFlow to classify images of forest fires and smoke.

Training of the model is performed using a custom classification layer added on top of the MobileNetV2 skeleton model. Training and validation data is housed in the 'dataset_forest' directory and preprocessed and augmented with ImageDataGenerator.

This code also uses the matplotlib and seaborn libraries to visualize the performance of the model.

## Installation

The following Python libraries need to be installed to run the project:

- Keras
- TensorFlow
- sklearn
- seaborn
- numpy
- matplotlib
- os

These libraries can be installed via pip:

```bash
pip install tensorflow keras sklearn seaborn numpy matplotlib os

## Usage
To start the training process, run the main.py file:

### python main.py

This will train the model, evaluate it on the validation set, and test it with images from the test set. It also generates a series of output images showing the predicted label over each test image.

To predict the label of an image, run the predict_image.py file:

### python predict_image.py

This will load the trained model and make a prediction on the specified image.
