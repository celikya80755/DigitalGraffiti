import cv2
import numpy as np

# Kamera öffnen (0 steht für die Standardkamera)
cap = cv2.VideoCapture(0)

# Initialisiere ein leeres Bild für die Spur des hellsten Punktes
trail_img = np.zeros((480, 640, 3), dtype=np.uint8)  # Passt zur Standard-Kameragröße (kann angepasst werden)

# Variable für die zuletzt gefundene Koordinate des hellsten Punktes
previous_point = None

while True:
    # Frame von der Kamera lesen
    ret, frame = cap.read()
    if not ret:
        break

    # Bild in Graustufen umwandeln
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Den hellsten Punkt im Bild finden
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

    # Den hellsten Punkt als Kreis markieren
    cv2.circle(frame, max_loc, 20, (0, 0, 255), 2)

    # Koordinaten auf dem Bild anzeigen
    cv2.putText(frame, f"Brightest Point: {max_loc}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Linie im Trail-Bild zeichnen, falls vorheriger Punkt existiert
    if previous_point is not None:
        cv2.line(trail_img, previous_point, max_loc, (0, 255, 0), 2)

    # Speichern der aktuellen Position als vorheriger Punkt für das nächste Frame
    previous_point = max_loc

    # Hauptbild und Spur anzeigen
    cv2.imshow('Hellster Punkt', frame)
    cv2.imshow('Spur des hellsten Punktes', trail_img)

    # Beenden, wenn 'q' gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera freigeben und Fenster schließen
cap.release()
cv2.destroyAllWindows()
