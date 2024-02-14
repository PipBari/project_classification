import numpy as np
import tensorflow as tf
from keras.models import load_model


class ImageClassifier:
    def __init__(self):
        self.model_path = 'D:/лабы/nei/my_model.h5'
        self.model = load_model(self.model_path)
        self.img_height = 512
        self.img_width = 512
        self.class_names_russian = {
            'healthy': 'здоров',
            'sick': 'болен',
            'undefined': 'неопределен'
        }
        try:
            self.class_names = self.model.class_names
        except AttributeError:
            self.class_names = ['healthy', 'sick', 'undefined']

    def classify_image(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((self.img_height, self.img_width))
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = self.class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        predicted_class_russian = self.class_names_russian.get(predicted_class, 'Неопределен')

        return predicted_class_russian, confidence
