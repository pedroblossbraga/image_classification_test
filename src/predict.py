from pyexpat import model
import argparse
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras

def model_inference(
                    model,
                    img_path,
                    img_height,
                    img_width,
                    class_names):
  # load image in desired shape
  img = tf.keras.utils.load_img(
      img_path, target_size=(img_height, img_width)
  )
    
  # transform to array
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  # predict
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  print(
      "This image most likely belongs to {} with a {:.3f}% confidence.".format(
          class_names[np.argmax(score)], 100 * np.max(score)
        )
  )
def lite_model_inference(
                    # model,
                    model,
                    img_path,
                    img_height,
                    img_width,
                    class_names):
  # load image in desired shape
  img = tf.keras.utils.load_img(
      img_path, target_size=(img_height, img_width)
  )
    
  # transform to array
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions_lite = model(sequential_1_input=img_array)['outputs']
  score_lite = tf.nn.softmax(predictions_lite)
  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
  )

def main(
        image_path,
        model_path = None,
        img_height = 180,
        img_width = 180,
        class_names = ['not st george', 'st george'],
        root_path = os.path.abspath(os.curdir)
    ):
    print(image_path)
    print('loading model...')
    if model_path is None:
        print('Using default path for model.')
        interpreter = tf.lite.Interpreter(model_path=os.path.join(root_path, 'model.tflite'))
        model = interpreter.get_signature_runner('serving_default')
        print('model loaded! \nPredicting class of image...')
    
    model_inference(
                    model = model,
                    img_path = image_path,
                    img_height = img_height,
                    img_width = img_width,
                    class_names = class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Image to predict')
    parser.add_argument('image_path', help='Enter the path of the image you would like to predict in.')
    args = parser.parse_args()

    main(args.image_path)
