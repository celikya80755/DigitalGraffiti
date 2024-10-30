import cv2
import numpy as np

# Kamera öffnen (0 steht für die Standardkamera)
cap = cv2.VideoCapture(0)

# Variable für die zuletzt gefundene Koordinate des hellsten Punktes
brightest_point = None

while True:
    # Frame von der Kamera lesen
    ret, frame = cap.read()
    if not ret:
        break

    # Bild in Graustufen umwandeln
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Den hellsten Punkt im Bild finden
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    brightest_point = max_loc  # Speichern der Koordinate des hellsten Punktes

    # Den hellsten Punkt als Kreis markieren
    cv2.circle(frame, max_loc, 20, (0, 0, 255), 2)

    # Koordinaten auf dem Bild anzeigen
    cv2.putText(frame, f"Brightest Point: {brightest_point}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                2)

    # Bild und Koordinate anzeigen
    cv2.imshow('Hellster Punkt', frame)

    # In einem separaten Fenster die Koordinate anzeigen
    coords_img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(coords_img, f"Koordinate: {brightest_point}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                2)
    cv2.imshow('Koordinaten Fenster', coords_img)

    # Beenden, wenn 'q' gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera freigeben und Fenster schließen
cap.release()
cv2.destroyAllWindows()
