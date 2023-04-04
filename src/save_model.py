import tensorflow as tf
import os

def convert_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    return tflite_model

def save_model(model, out_dir,
               model_name = 'model.tflite'):
    with open(os.path.join(out_dir, model_name), 'wb') as f:
        f.write(model)