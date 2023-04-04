from logging import root
from setup_dir import setup_dir
from models import Model
from plots import plot_history_results
from save_model import convert_model, save_model

import os

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def main(
    root_path = os.path.abspath(os.curdir),
    
    # loader parameters
    batch_size = 32,
    img_height = 180,
    img_width = 180,
    epochs = 10,
    show_history = True,
    create_setup =  False
):
    """
    ------------------------
    Parameters
    show_history: bool; plot history or not.
    create_setup: bool; create folders, extract and store images or 
        not -> if you already have the "image_dataset/not st george"
        and "image_dataset/st george" directories populated with 
        images, you can set create_setup to False.
    ------------------------
    """
    if create_setup:
        # load images into new folder
        setup_dir(root_path = root_path)
    
    # sample training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(root_path, 'image_dataset'),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    # sample validation data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(root_path, 'image_dataset'),
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    class_names = train_ds.class_names

    # prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # instance of classifier
    model = Model(
        img_height=img_height,
        img_width=img_width,
        num_classes=len(class_names)
    ).get_model()
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    # traing the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    if show_history:
        plot_history_results(history, epochs)
        
    
    save_model(model = model,
               out_dir = root_path,
               model_name = 'model.h5')
    model.save(os.path.join(root_path, 'my_model.h5'))
    
    # convert model to tflite_model
    tflite_model = convert_model(model)
    save_model(model = tflite_model,
               out_dir = root_path,
               model_name = 'model.tflite')
    

if __name__ == "__main__":
    main()