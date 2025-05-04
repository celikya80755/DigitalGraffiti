import ctypes
import cv2
import numpy as np
import screeninfo
from kalman_filter import KalmanFilter

class DigitalGraffiti:

    # main settings
    SCREEN_ID = 1
    CURRENT_CAM = 1
    MIRRORED = False
    KALMAN = False

    # defaults
    DEFAULT_THRESHOLD = 120
    DEFAULT_COLOR = (0, 0, 255)

    # spray settings
    SPRAY_OPACITY = 0.5
    DENSITY = 70
    RADIUS = 10

    # performance
    DOWNSCALE_FACTOR = 1.5

    # color map
    COLOR_MAP = [
        (255, 255, 255),  # white
        (255, 1, 132),  # purple
        (255, 1, 31),  # blue
        (255, 240, 1),  # cyan
        (43, 255, 1),  # green
        (1, 246, 255),  # yellow
        (1, 150, 255),  # orange
        (1, 1, 255)  # red
    ]

    # color timer
    COLOR_TIMER = 0

    # color picker constraints
    BORDER = 1815
    CIRCLE_PAD_TOP = 49
    CIRCLE_PAD_LEFT = 28
    CIRCLE_RADIUS = 68
    CIRCLE_PAD = 21
    CIRCLE_COUNT = 8

    # scren size
    screen = screeninfo.get_monitors()[SCREEN_ID]
    WINDOW_WIDTH, WINDOW_HEIGHT = int(screen.width / DOWNSCALE_FACTOR), int(screen.height / DOWNSCALE_FACTOR)

    # texture scale
    TEXTURE_WIDTH = 2000
    TEXTURE_HEIGHT = 1333
    TEXTURE_WIDTH_SCALE = WINDOW_WIDTH / TEXTURE_WIDTH
    TEXTURE_HEIGHT_SCALE = WINDOW_HEIGHT / TEXTURE_HEIGHT

    def __init__(self):
        self.kalman = KalmanFilter()
        self.capture = cv2.VideoCapture(self.CURRENT_CAM)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.WINDOW_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.WINDOW_HEIGHT)

        # Load the texture image and resize to the screen size
        texture = cv2.imread('texture_colors.jpg', cv2.IMREAD_COLOR)
        if texture is None:
            print("Could not load texture_colors.jpg. Make sure the file exists.")
            texture = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
        else:
            texture = cv2.resize(texture, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

        # Set the canvas to the texture
        self.canvas = texture.copy()

        self.buffer = np.full((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), 0, dtype=np.uint8)

        cv2.namedWindow('Kamerafeed', cv2.WINDOW_NORMAL)  # Fenster dynamisch skalierbar
        cv2.namedWindow('Graffiti', cv2.WINDOW_NORMAL)

        cv2.moveWindow('Graffiti', self.screen.x-1, self.screen.y-1)
        cv2.setWindowProperty('Graffiti', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow('Graffiti', self.resize_canvas('Graffiti', self.canvas))

        self.threshold = self.DEFAULT_THRESHOLD
        self.current_color = self.DEFAULT_COLOR  # Standardfarbe: Rot
        self.create_control_sliders()

        # Führt die Kalibrierung aus
        self.transformation_matrix = self.calibrate_perspective()

        # Starte die Hauptkamera-Schleife
        self.camera_loop()

    def create_control_sliders(self):
        cv2.createTrackbar('R', 'Kamerafeed', 0, 255, self.update_color)
        cv2.createTrackbar('G', 'Kamerafeed', 0, 255, self.update_color)
        cv2.createTrackbar('B', 'Kamerafeed', 0, 255, self.update_color)
        cv2.createTrackbar('Threshold', 'Kamerafeed', 0, 255, self.update_threshold)

    def calibrate_perspective(self):
        points = []
        cv2.circle(self.canvas, (10, 10), 10, (255,255,255))
        cv2.circle(self.canvas, (self.WINDOW_WIDTH-10, 10), 10, (255, 255, 255))
        cv2.circle(self.canvas, (10, self.WINDOW_HEIGHT-10), 10, (255, 255, 255))
        cv2.circle(self.canvas, (self.WINDOW_WIDTH-10, self.WINDOW_HEIGHT-10), 10, (255, 255, 255))

        cv2.putText(self.canvas, "1", (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.canvas, "2", (self.WINDOW_WIDTH - 20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.canvas, "3", (20, self.WINDOW_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.canvas, "4", (self.WINDOW_WIDTH - 20, self.WINDOW_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        cv2.imshow('Graffiti', self.resize_canvas('Graffiti', self.canvas))

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                print(f"Punkt hinzugefügt: {x}, {y}")
                if len(points) == 4:
                    cv2.destroyWindow('Kalibrierung')

        print("Klicke auf vier Punkte für die Perspektivkorrektur (oben links, oben rechts, unten rechts, unten links)")
        while True:
            successful, frame = self.capture.read()
            if not successful:
                print("Fehler beim Lesen der Kamera während der Kalibrierung!")
                exit()

            for idx, point in enumerate(points):
                cv2.circle(frame, point, 5, (0, 255, 0), -1)
                cv2.putText(frame, f"{idx + 1}", (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('Kalibrierung', frame)
            cv2.setMouseCallback('Kalibrierung', click_event)

            if len(points) == 4:
                break
            if cv2.waitKey(1) & 0xFF == 27:
                print("Kalibrierung abgebrochen.")
                exit()

        print("Kalibrierung abgeschlossen: ", points)

        dst_points = np.array([[0, 0], [self.WINDOW_WIDTH - 1, 0],
                               [0, self.WINDOW_HEIGHT - 1],
                               [self.WINDOW_WIDTH - 1, self.WINDOW_HEIGHT - 1]
                               ], dtype="float32")
        src_points = np.array(points, dtype="float32")
        self.canvas = cv2.resize(self.canvas, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.clear_canvas()
        return cv2.getPerspectiveTransform(src_points, dst_points)

    def apply_transformation(self, frame):
        return cv2.warpPerspective(frame, self.transformation_matrix, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

    def resize_canvas(self, window_name, image):
        try:
            # Überprüfen, ob das Fenster sichtbar ist
            if not cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
                raise ValueError(f"Fenster '{window_name}' existiert nicht oder ist nicht sichtbar.")

            # Fenstergröße abrufen
            rect = cv2.getWindowImageRect(window_name)
            width, height = rect[2], rect[3]

            # Validierung der Breite und Höhe
            if width <= 0 or height <= 0:
                print(f"Ungültige Fenstergröße ({width}, {height}). Verwende Bildgröße.")
                width, height = image.shape[1], image.shape[0]  # Standard auf Bildgröße setzen

            # Bild skalieren
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Fehler beim Skalieren des Bildes: {e}")
            return image  # Rückgabe des Originalbildes, falls Fehler auftreten

    def camera_loop(self):
        while True:
            successful, video_frame = self.capture.read()
            if not successful:
                try:
                    ctypes.windll.user32.MessageBoxW(None, u"No webcam found!", u"Error", 0)
                except AttributeError:
                    print("No webcam found!")
                break

            transformed_frame = self.apply_transformation(video_frame)
            brightest_point_value, brightest_point_location = self.find_brightest_point(transformed_frame)

            if self.MIRRORED:
                brightest_point_location = self.mirror_point_horizontally(brightest_point_location)
            else:
                brightest_point_location = brightest_point_location

            if brightest_point_value >= self.threshold:
                if (self.KALMAN):
                    predicted_point = self.kalman.apply_kalman(brightest_point_location)
                else:
                    predicted_point = brightest_point_location
                self.show_brightest_point(transformed_frame, predicted_point)
                self.handle_point(self.canvas, predicted_point, self.RADIUS, self.current_color)

            self.show_brightest_point_text(transformed_frame, brightest_point_location)
            self.show_color_options(transformed_frame)
            cv2.imshow('Kamerafeed', self.resize_canvas('Kamerafeed', transformed_frame))
            cv2.imshow('Graffiti', self.resize_canvas('Graffiti', self.canvas))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_color = self.COLOR_MAP[7]
            elif key == ord('g'):
                self.current_color = self.COLOR_MAP[4]
            elif key == ord('b'):
                self.current_color = self.COLOR_MAP[2]
            elif key == ord('y'):
                self.current_color = self.COLOR_MAP[5]
            elif key == ord('c'):
                self.clear_canvas()

        self.close()

    def update_color(self, x):
        r = cv2.getTrackbarPos('R', 'Kamerafeed')
        g = cv2.getTrackbarPos('G', 'Kamerafeed')
        b = cv2.getTrackbarPos('B', 'Kamerafeed')
        self.current_color = (b, g, r)

    def update_threshold(self, x):
        self.threshold = cv2.getTrackbarPos('Threshold', 'Kamerafeed')

    def mirror_point_horizontally(self, point_location):
        return (self.WINDOW_WIDTH - point_location[0]), point_location[1]

    def find_brightest_point(self, video_frame):
        grayscale_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        _, brightest_point_value, _, brightest_point_location = cv2.minMaxLoc(grayscale_image)
        return brightest_point_value, brightest_point_location

    def show_brightest_point(self, video_frame, predicted_point):
        cv2.circle(video_frame, predicted_point, 20, self.current_color, 2)

    def handle_point(self, canvas, center, radius, color):
        if center[0] > (self.BORDER * self.TEXTURE_WIDTH_SCALE):
            color_index = self.get_circle_index(center[0], center[1])
            if color_index >= 0:
                self.COLOR_TIMER += 1
                print(f"Detected color {color_index} for {self.COLOR_TIMER} frames")
            else:
                print(f"Resetting color timer to 0")
                self.COLOR_TIMER = 0

            if self.COLOR_TIMER >= 60:
                self.current_color = self.COLOR_MAP[color_index]

        else:
            self.spray_on_canvas(self, canvas, center, radius, color)

    def spray_on_canvas(self, canvas, center, radius, color):
        spray_buffer = np.full((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), 255, dtype=np.uint8)

        # Draw random spray dots
        for _ in range(self.DENSITY):
            x_offset = np.random.randint(-radius, radius)
            y_offset = np.random.randint(-radius, radius)
            if x_offset ** 2 + y_offset ** 2 <= radius ** 2:
                cv2.circle(spray_buffer, (center[0] + x_offset, center[1] + y_offset), 1, color, -1)

        # Convert images to float for blending
        canvas_float = canvas.astype(np.float32) / 255.0
        spray_float = spray_buffer.astype(np.float32) / 255.0

        # Create an alpha mask where the spray is applied
        mask = (spray_buffer != 255).any(axis=2).astype(np.float32) * self.SPRAY_OPACITY
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Blend the spray onto the canvas
        blended = canvas_float * (1 - mask) + spray_float * mask
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

        # Update the canvas
        self.canvas = blended

    def get_circle_index(self, x, y):
        for i in range(self.CIRCLE_COUNT):
            cx = (self.BORDER + self.CIRCLE_PAD_LEFT + self.CIRCLE_RADIUS) * self.TEXTURE_WIDTH_SCALE
            cy = (self.CIRCLE_PAD_TOP + i * (
                        self.CIRCLE_PAD + 2 * self.CIRCLE_RADIUS) + self.CIRCLE_RADIUS) * self.TEXTURE_HEIGHT_SCALE

            dx = x - cx
            dy = y - cy

            rx = self.CIRCLE_RADIUS * self.TEXTURE_WIDTH_SCALE
            ry = self.CIRCLE_RADIUS * self.TEXTURE_HEIGHT_SCALE

            # Punkt-in-Ellipse-Prüfung
            if (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) < 1:
                print(f"Color picker detected: {i}")
                return i

        return -1

    def show_brightest_point_text(self, video_frame, brightest_point):
        cv2.putText(video_frame, f"Brightest Point: {brightest_point}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    def show_color_options(self, video_frame):
        cv2.putText(video_frame, "Farbauswahl: [R] Rot | [G] Grün | [B] Blau | [Y] Gelb",
                    (10, video_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def clear_canvas(self):
        # Reset the canvas to the original texture
        texture = cv2.imread('texture_colors.jpg', cv2.IMREAD_COLOR)
        if texture is not None:
            texture = cv2.resize(texture, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            self.canvas = texture.copy()
        else:
            self.canvas.fill(0)
        self.buffer.fill(0)

    def close(self):
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    graffiti_app = DigitalGraffiti()