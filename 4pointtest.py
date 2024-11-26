import cv2
import numpy as np

# Globale Variablen für die Mausklicks
points = []

def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Punkt hinzugefügt: {x}, {y}")

        # Wenn 4 Punkte ausgewählt wurden, starte die Transformation
        if len(points) == 4:
            apply_transformation()

def apply_transformation():
    global points
    # Zielgröße für die Transformation
    width, height = 400, 300
    dst_points = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype="float32")

    # Berechnung der Perspektivtransformation
    src_points = np.array(points, dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Transformation auf den letzten Frame anwenden
    transformed = cv2.warpPerspective(frame, matrix, (width, height))

    # Transformiertes Bild anzeigen
    cv2.imshow("Transformierte Perspektive", transformed)

    # Punkte zurücksetzen, um neue Transformation zu ermöglichen
    points.clear()

# Webcam öffnen
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Fehler beim Zugriff auf die Kamera.")
    exit()

cv2.namedWindow("Live-Feed")
cv2.setMouseCallback("Live-Feed", select_points)

print("Klicke auf vier Punkte im Live-Feed, um die Perspektive zu transformieren.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Lesen des Kamerabilds.")
        break

    # Original-Feed anzeigen
    cv2.imshow("Live-Feed", frame)

    # Warten auf Tastendruck
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC zum Beenden
        break

# Resourcen freigeben
cap.release()
cv2.destroyAllWindows()
