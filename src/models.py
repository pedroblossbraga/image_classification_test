import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class Model:
    def __init__(self,
                 img_height,
                 img_width,
                 num_classes):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
    
    def get_model(self):
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal",
                                input_shape=(self.img_height,
                                            self.img_width,
                                            3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
            )
        return Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, name="outputs")
            ])
