from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import img_to_array, load_img
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

# Dataset and output paths
dataset_path = 'C:/Users/Bora/Downloads/dataset_forest/train'
test_folder = 'C:/Users/Bora/Downloads/dataset_forest/test_small'
output_folder = 'C:/Users/Bora/Downloads/dataset_forest/test_output'

os.makedirs(output_folder, exist_ok=True)

# Data preprocessing and augmentation
data_gen = ImageDataGenerator(rescale=1. / 255,
                              horizontal_flip=True,
                              validation_split=0.2)

# Train and validation dataset
train_data = data_gen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary',
                                          subset='training')
val_data = data_gen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary',
                                        subset='validation')



def train_and_evaluate_model(train_data, val_data):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(1, activation="sigmoid")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"])

    H = model.fit(train_data, validation_data=val_data, epochs=10)  # Epoch sayısını artırdık.
    return model, H
def test_and_visualize_model(model, test_folder, output_folder):
    test_images = os.listdir(test_folder)
    y_true = []
    y_pred = []
    for img_name in test_images:
        img_path = os.path.join(test_folder, img_name)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        prediction = model.predict(img_array)
        prediction_label = 'smoke' if prediction > 0.5 else 'fire'
        if 'smoke' in img_name or 'fume' in img_name:
            y_true.append('smoke')
        elif 'fire' in img_name:
            y_true.append('fire')
        else:
            print(f"Unexpected image name: {img_name}")
            continue
        y_pred.append(prediction_label)
        plt.imshow(img)
        plt.title(prediction_label)
        plt.axis('off')
        output_path = os.path.join(output_folder, img_name)
        plt.savefig(output_path)
        plt.clf()
    print(classification_report(y_true, y_pred, target_names=['fire', 'smoke']))
    cm = confusion_matrix(y_true, y_pred, labels=['fire', 'smoke'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.show()


# Modeli eğit ve değerlendir
model, H = train_and_evaluate_model(train_data, val_data)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(H.history['accuracy'], label='Training Accuracy')
plt.plot(H.history['val_accuracy'], label='Validation Accuracy')
plt.title('Epochs vs. Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(H.history['loss'], label='Training Loss')
plt.plot(H.history['val_loss'], label='Validation Loss')
plt.title('Epochs vs. Training and Validation Loss')
plt.legend()

plt.show()

# Modeli test et ve sonuçları görselleştir
test_and_visualize_model(model, test_folder, output_folder)