import numpy as np
from keras.models import model_from_json


class FaceRecognition:
    '''
    Creates the model to predict an emotion based on a camera capture
    '''
    # EMOTIONS = ['Angry', 'Disgust''Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] # Model 1
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"] # Model 2

    def __init__(self, ser_file, weights):
        '''
        Initialize the model loading it from a serialized file.

        :param ser_file: Serialized file created from the model
        :param weights: Weights in the model
        '''
        with open(ser_file, 'r') as json_file:
            # Load the model from a serialized file
            model_json = json_file.read()
            self.model = model_from_json(model_json)

        # Once we loaded we can use the function defined in the jupyter file
        self.model.load_weights(weights)
        self.model._make_predict_function()
        self.prediction = ''

    def predict_emotion(self, img):
        '''
        Predicts the emotion based on an image.

        :param img: Image with the facial emotion. Usually a frame from a video capture
        :return: prediction
        '''
        self.prediction = self.model.predict(img)
        return FaceRecognition.EMOTIONS[np.argmax(self.prediction)]
