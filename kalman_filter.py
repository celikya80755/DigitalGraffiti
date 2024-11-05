import cv2
import numpy as np

class KalmanFilter:

    def __init__(self):
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)

        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)

        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32) * 0.03

        self.kalman.measurementNoiseCov = np.array([[1, 0],
                                               [0, 1]], np.float32) * 1

    def apply_kalman(self, max_loc):
        measurement = np.array([[np.float32(max_loc[0])],
                                [np.float32(max_loc[1])]], np.float32)
        self.kalman.correct(measurement)

        predicted = self.kalman.predict()
        predicted_point = (int(predicted[0]), int(predicted[1]))
        return predicted_point