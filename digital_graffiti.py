import ctypes

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from kalman_filter import KalmanFilter

class DigitalGraffiti:

    threshold_value = 200

    def __init__(self):
        self.kalman = KalmanFilter()

        self.capture = cv2.VideoCapture(0)
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.buffer = np.zeros((480, 640, 3), dtype=np.uint8)
        self.create_gui()
        self.camera_loop()

    def create_gui(self):
        window = tk.Tk()
        window.title("Digital Graffiti")

        label = tk.Label(window)
        label.pack()

    def camera_loop(self):
        while True:
            successful, video_frame = self.capture.read()
            if not successful:
                ctypes.windll.user32.MessageBoxW(None, u"No webcam found!", u"Error", 0)
                break

            brightest_point_value, brightest_point_location = self.find_brightest_point(video_frame)
            if brightest_point_value >= self.threshold_value:
                predicted_point = self.kalman.apply_kalman(brightest_point_location)
                self.show_brightest_point(video_frame, predicted_point)
                self.spray_on_canvas(self.canvas, predicted_point, 10, (0, 0, 255))

            self.show_brightest_point_text(video_frame, brightest_point_location)

            self.update_window('Hellster Punkt', video_frame)
            self.update_window('Graffiti', self.canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.close()

    def find_brightest_point(self, video_frame):
        """Finds the brightest point in an image and returns its value and its location."""

        grayscale_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(grayscale_image, self.threshold_value, 255, cv2.THRESH_BINARY)
        _, brightest_point_value, _, brightest_point_location = cv2.minMaxLoc(grayscale_image, threshold_image)
        return brightest_point_value, brightest_point_location

    def show_brightest_point(self, video_frame, predicted_point):
        """Creates a circle in the webcam preview showing the brightest detected point."""

        cv2.circle(video_frame, predicted_point, 20, (0, 0, 255, 10), 2)

    def spray_on_canvas(self, canvas, center, radius, color):
        """ Adds a spray effect at the given point, radius and color in a low opacity, getting brighter
        the longer a position is held."""

        opacity = 0.05

        for _ in range(100):
            x_offset = np.random.randint(-radius, radius)
            y_offset = np.random.randint(-radius, radius)
            if x_offset ** 2 + y_offset ** 2 <= radius ** 2:
                cv2.circle(self.buffer, (center[0] + x_offset, center[1] + y_offset), 1, color, -1)

        self.canvas = cv2.addWeighted(self.buffer, opacity, canvas, 1 - opacity, 0)

    def show_brightest_point_text(self, video_frame, brightest_point):
        """Debug function to show the coordinates of the currently detected point."""

        cv2.putText(video_frame, f"Brightest Point: {brightest_point}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255),2)

    def update_window(self, title, window):
        """Updates the window displaying an image object."""

        cv2.imshow(title, window)

    def close(self):
        """Closes windows and releases webcam."""

        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    graffiti_app = DigitalGraffiti()