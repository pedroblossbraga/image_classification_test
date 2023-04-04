import tensorflow as tf
import numpy as np

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