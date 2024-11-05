import cv2
import numpy as np

from kalman_filter import KalmanFilter


class DigitalGraffiti:

    threshold_value = 200

    def __init__(self):
        self.kalman = KalmanFilter()

        self.capture = cv2.VideoCapture(0)
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera_loop()

    def camera_loop(self):
        while True:
            successful, video_frame = self.capture.read()
            if not successful:
                break

            _, brightest_point_value, _, brightest_point_location = self.find_brightest_point(video_frame)
            if brightest_point_value >= self.threshold_value:

                predicted_point = self.kalman.apply_kalman(brightest_point_location)
                self.show_brightest_point(video_frame, predicted_point)
                self.spray_on_canvas(self.canvas, predicted_point, 10, (0, 0, 255))

            self.show_brightest_point_text(video_frame, brightest_point_location)
            self.update_window('Hellster Punkt', video_frame)
            self.update_window('Graffiti', self.canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def find_brightest_point(self, video_frame):
        grayscale_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(grayscale_image, self.threshold_value, 255, cv2.THRESH_BINARY)
        return cv2.minMaxLoc(grayscale_image, threshold_image);

    def show_brightest_point(self, video_frame, predicted_point):
        cv2.circle(video_frame, predicted_point, 20, (0, 0, 255), 2)

    def spray_on_canvas(self, canvas, center, radius, color):
        for _ in range(100):  # Anzahl der Sprühpunkte
            x_offset = np.random.randint(-radius, radius)
            y_offset = np.random.randint(-radius, radius)
            if x_offset ** 2 + y_offset ** 2 <= radius ** 2:
                cv2.circle(canvas, (center[0] + x_offset, center[1] + y_offset), 1, color, -1)

    def show_brightest_point_text(self, video_frame, brightest_point):
        cv2.putText(video_frame, f"Brightest Point: {brightest_point}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255),2)

    def update_window(self, title, window):
        cv2.imshow(title, window)

    def cleanup(self):
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    graffiti_app = DigitalGraffiti()