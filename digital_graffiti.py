import ctypes
import cv2
import numpy as np
from kalman_filter import KalmanFilter

class DigitalGraffiti:
    threshold_value = 200

    def __init__(self):
        self.kalman = KalmanFilter()
        self.capture = cv2.VideoCapture(0)
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.buffer = np.zeros((480, 640, 3), dtype=np.uint8)
        self.current_color = (0, 0, 255)  # Standardfarbe: Rot

        # Starte die Hauptkamera-Schleife
        self.camera_loop()

    def camera_loop(self):
        while True:
            successful, video_frame = self.capture.read()
            if not successful:
                try:
                    ctypes.windll.user32.MessageBoxW(None, u"No webcam found!", u"Error", 0)
                except AttributeError:
                    print("No webcam found!")
                break

            # Helligsten Punkt finden und malen
            brightest_point_value, brightest_point_location = self.find_brightest_point(video_frame)
            if brightest_point_value >= self.threshold_value:
                predicted_point = self.kalman.apply_kalman(brightest_point_location)
                self.show_brightest_point(video_frame, predicted_point)
                self.spray_on_canvas(self.canvas, predicted_point, 10, self.current_color)

            # Text anzeigen und Kamera-Frames aktualisieren
            self.show_brightest_point_text(video_frame, brightest_point_location)
            self.show_color_options(video_frame)
            self.update_window('Hellster Punkt', video_frame)
            self.update_window('Graffiti', self.canvas)

            # Farbauswahl durch Tastenanschläge
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Programm beenden
                break
            elif key == ord('r'):  # Rot
                self.current_color = (0, 0, 255)
            elif key == ord('g'):  # Grün
                self.current_color = (0, 255, 0)
            elif key == ord('b'):  # Blau
                self.current_color = (255, 0, 0)
            elif key == ord('y'):  # Gelb
                self.current_color = (0, 255, 255)
            elif key == ord('c'):
                self.clear_canvas()

        self.close()

    def find_brightest_point(self, video_frame):
        grayscale_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(grayscale_image, self.threshold_value, 255, cv2.THRESH_BINARY)
        _, brightest_point_value, _, brightest_point_location = cv2.minMaxLoc(grayscale_image, threshold_image)
        return brightest_point_value, brightest_point_location

    def show_brightest_point(self, video_frame, predicted_point):
        cv2.circle(video_frame, predicted_point, 20, self.current_color, 2)

    def spray_on_canvas(self, canvas, center, radius, color):
        opacity = 0.05
        for _ in range(100):
            x_offset = np.random.randint(-radius, radius)
            y_offset = np.random.randint(-radius, radius)
            if x_offset ** 2 + y_offset ** 2 <= radius ** 2:
                cv2.circle(self.buffer, (center[0] + x_offset, center[1] + y_offset), 1, color, -1)
        self.canvas = cv2.addWeighted(self.buffer, opacity, canvas, 1 - opacity, 0)

    def show_brightest_point_text(self, video_frame, brightest_point):
        cv2.putText(video_frame, f"Brightest Point: {brightest_point}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    def show_color_options(self, video_frame):
        """Zeigt die verfügbaren Farboptionen im Kamerafenster an."""
        cv2.putText(video_frame, "Farbauswahl: [R] Rot | [G] Grün | [B] Blau | [Y] Gelb",
                    (10, video_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def update_window(self, title, window):
        cv2.imshow(title, window)

    def clear_canvas(self):
        self.canvas.fill(0)
        self.buffer.fill(0)
    def close(self):
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    graffiti_app = DigitalGraffiti()
