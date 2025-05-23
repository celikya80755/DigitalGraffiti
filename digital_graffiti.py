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
    DENSITY = 150
    radius = 10

    # performance
    DOWNSCALE_FACTOR = 1.5

    # color map
    COLOR_MAP = [
        (255, 255, 255),  # 0 white
        (255, 1, 31),  # 1 blue
        (43, 255, 1),  # 2 green
        (1, 246, 255),  # 3 yellow
        (1, 1, 255)  # 4 red
    ]

    BRUSH_MAP = [
        5, 10, 15
    ]

    DENSITY_MAP = [
        100, 200, 300]

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

    active_index = -1
    timer_index = -1
    timer_value = 0

    def __init__(self):
        self.kalman = KalmanFilter()
        self.capture = cv2.VideoCapture(self.CURRENT_CAM)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.WINDOW_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.WINDOW_HEIGHT)

        self.setup_canvas()

    def setup_canvas(self):
        # Load the texture image and resize to the screen size
        texture = cv2.imread('texture_colors_sizes.jpg', cv2.IMREAD_COLOR)
        if texture is None:
            print("Could not load texture_colors_sizes.jpg. Make sure the file exists.")
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

        self.spray_mask = self.generate_spray_mask(self.radius, self.DENSITY)

        # Führt die Kalibrierung aus
        self.transformation_matrix = self.calibrate_perspective()

        # Starte die Hauptkamera-Schleife
        self.camera_loop()

    def generate_spray_mask(self, radius, density):
        mask = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        center = radius
        for _ in range(density):
            x_offset = np.random.randint(-radius, radius)
            y_offset = np.random.randint(-radius, radius)
            if x_offset ** 2 + y_offset ** 2 <= radius ** 2:
                mask[center + y_offset, center + x_offset] = 255
        return mask

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
                self.handle_point(self.canvas, predicted_point, self.radius, self.current_color)

            self.show_brightest_point_text(transformed_frame, brightest_point_location)
            self.show_color_options(transformed_frame)
            cv2.imshow('Kamerafeed', self.resize_canvas('Kamerafeed', transformed_frame))
            cv2.imshow('Graffiti', self.resize_canvas('Graffiti', self.canvas))

            self.draw_selection_progress()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.close()
            elif key == ord('r'):
                self.current_color = self.COLOR_MAP[4]
            elif key == ord('g'):
                self.current_color = self.COLOR_MAP[2]
            elif key == ord('b'):
                self.current_color = self.COLOR_MAP[1]
            elif key == ord('y'):
                self.current_color = self.COLOR_MAP[3]
            elif key == ord('c'):
                self.clear_canvas()
            elif key == ord('p'):
                cv2.destroyAllWindows()
                break

        self.setup_canvas()

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
                if not hasattr(self, "current_hover_index") or self.current_hover_index != color_index:
                    self.COLOR_TIMER = 0  # Neue Auswahl -> Timer zurücksetzen
                self.current_hover_index = color_index
                self.COLOR_TIMER += 1
                print(f"Detected color {color_index} for {self.COLOR_TIMER} frames")
            else:
                self.COLOR_TIMER = 0
                self.current_hover_index = None
                return  # Keine Auswahl

            if self.COLOR_TIMER >= 30:
                if color_index < len(self.COLOR_MAP):
                    self.current_color = self.COLOR_MAP[color_index]
                else:
                    brush_index = color_index - len(self.COLOR_MAP)
                    self.radius = self.BRUSH_MAP[brush_index]
                    self.DENSITY = self.DENSITY_MAP[brush_index]
                    self.spray_mask = self.generate_spray_mask(self.radius, self.DENSITY)
                self.last_selected_index = color_index
                self.COLOR_TIMER = 30  # Festhalten

        else:
            self.COLOR_TIMER = 0
            self.current_hover_index = None
            self.spray_on_canvas(canvas, center, radius, color)

    def draw_selection_progress(self):
        overlay = self.canvas.copy()

        if self.COLOR_TIMER >= 30:
            indices = [self.last_selected_index] if hasattr(self, "last_selected_index") else []
        elif self.COLOR_TIMER > 0 and hasattr(self, "current_hover_index"):
            indices = [self.current_hover_index]
        else:
            indices = []

        for i in range(self.CIRCLE_COUNT):
            # Zentrum berechnen (wie in get_circle_index)
            cx = int((self.BORDER + self.CIRCLE_PAD_LEFT + self.CIRCLE_RADIUS) * self.TEXTURE_WIDTH_SCALE)
            cy = int((self.CIRCLE_PAD_TOP + i * (
                        self.CIRCLE_PAD + 2 * self.CIRCLE_RADIUS) + self.CIRCLE_RADIUS) * self.TEXTURE_HEIGHT_SCALE)

            # Skaliertes Ellipsen-Radius
            rx = int(self.CIRCLE_RADIUS * self.TEXTURE_WIDTH_SCALE)
            ry = int(self.CIRCLE_RADIUS * self.TEXTURE_HEIGHT_SCALE)

            if i in indices:
                if self.COLOR_TIMER < 30:
                    progress = self.COLOR_TIMER / 30.0
                    end_angle = int(360 * progress)
                    cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, end_angle, (255, 255, 255), 2)
                else:
                    cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, (255, 255, 255), 2)

        cv2.imshow('Graffiti', self.resize_canvas('Graffiti', overlay))

    def spray_on_canvas(self, canvas, center, radius, color):
        x, y = center
        mask = self.spray_mask

        # Get region of interest
        h, w = mask.shape
        x1, y1 = max(x - radius, 0), max(y - radius, 0)
        x2, y2 = min(x + radius + 1, self.WINDOW_WIDTH), min(y + radius + 1, self.WINDOW_HEIGHT)

        roi_canvas = canvas[y1:y2, x1:x2]
        roi_mask = mask[(y1 - y + radius):(y2 - y + radius), (x1 - x + radius):(x2 - x + radius)]

        # Create spray image with the selected color
        spray_color = np.zeros_like(roi_canvas)
        spray_color[roi_mask > 0] = color

        # Blend
        alpha = self.SPRAY_OPACITY
        mask3 = (roi_mask > 0).astype(np.float32)[..., np.newaxis]
        roi_canvas[:] = (roi_canvas * (1 - mask3 * alpha) + spray_color * (mask3 * alpha)).astype(np.uint8)

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
        texture = cv2.imread('texture_colors_sizes.jpg', cv2.IMREAD_COLOR)
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