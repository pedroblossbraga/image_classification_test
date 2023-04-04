import matplotlib.pyplot as plt

def plot_history_results(history, epochs):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = list(range(epochs))

  plt.figure(figsize=(10, 3))
  plt.subplot(1,2,1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='best')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1,2,2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='best')
  plt.title('Training and Validation Loss')

  plt.tight_layout()
  plt.show()