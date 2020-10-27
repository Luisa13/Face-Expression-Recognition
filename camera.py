import cv2
from model import FaceRecognition
import logging
import numpy as np
import os

json_file = r'.\model\serialization\model.json'
weights = r'.\model\weights.h5'
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img_size = 48


class Camera:
    '''
    Creates a camera object to process the frames in a video,
    detect a face and classify emotions.

    Args:
        :param video_capture:
        :param model:

    '''

    def __init__(self):
        '''
        Contructor.
        '''
        self.video_capture = cv2.VideoCapture(0)
        # self.classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if not os.path.exists(json_file):
            logging.error("The file does not exist")
        else:
            self.model = FaceRecognition(json_file, weights)
        # TODO logging parameters

    def __del__(self):
        '''
        Release the video capture

        :return:
        '''
        self.video_capture.release()

    def run(self):
        '''
        DEPRECATED (only used for debbug purposes and pc desk application)

        Goes through all the frames in the video capture and apply
        the pretrained model to make the prediction showing the image

        :return:
        '''
        while True:
            # Capture frame-by-frame
            ret, frame = self.video_capture.read()
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = classifier.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                logging.exception("\tNo faces found")
            else:
                for (x, y, w, h) in faces:
                    face_detected = gray[y:y + h, x:x + w]
                    roi = cv2.resize(face_detected, (img_size, img_size))
                    # Prediction
                    pred_emotion = self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

                    # Draw the rectangle wrapping the face
                    start_point = (x, y)
                    end_point = (x + w, y + h)
                    image = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 3)
                    image = cv2.putText(image, pred_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Display the resulting frame
                cv2.imshow('frame', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

    def get_frames(self):
        '''
        Gets all the frames took by the camera after applying
        the prediction from the model and return it

        :return: Image bytes
        '''
        # Capture frame-by-frame
        ret, frame = self.video_capture.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            logging.exception("\tNo faces found")
        else:
            for (x, y, w, h) in faces:
                face_detected = gray[y:y + h, x:x + w]
                roi = cv2.resize(face_detected, (img_size, img_size))
                # Prediction
                pred_emotion = self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

                # Draw the rectangle wrapping the face
                start_point = (x, y)
                end_point = (x + w, y + h)
                image = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 3)
                image = cv2.putText(image, pred_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                frame = image

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
