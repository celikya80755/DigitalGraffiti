import cv2
import numpy as np

# Kalman-Filter initialisieren
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

kalman.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], np.float32) * 1

# VideoCapture initialisieren
cap = cv2.VideoCapture(0)
trail_img = np.zeros((480, 640, 3), dtype=np.uint8)  # Passt zur Standard-Kameragröße (kann angepasst werden)

previous_point = None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold_value = 230
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray, thresholded)

    if max_val >= threshold_value:
        # Kalman-Filter aktualisieren
        measurement = np.array([[np.float32(max_loc[0])],
                                [np.float32(max_loc[1])]], np.float32)
        kalman.correct(measurement)

        # Schätzung des Kalman-Filters
        predicted = kalman.predict()
        predicted_point = (int(predicted[0]), int(predicted[1]))

        # Kreise zeichnen
        cv2.circle(frame, predicted_point, 20, (0, 0, 255), 2)

    # Koordinaten auf dem Bild anzeigen
    cv2.putText(frame, f"Brightest Point: {max_loc}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Linie im Trail-Bild zeichnen, falls vorheriger Punkt existiert
    if previous_point is not None:
        cv2.line(trail_img, previous_point, max_loc, (0, 255, 0), 2)

    previous_point = max_loc

    # Hauptbild und Spur anzeigen
    cv2.imshow('Hellster Punkt', frame)
    cv2.imshow('Spur des hellsten Punktes', trail_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
