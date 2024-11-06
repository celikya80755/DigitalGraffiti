import cv2
import numpy as np

def update_color(x):
    global img
    r = cv2.getTrackbarPos('R', 'Image')
    g = cv2.getTrackbarPos('G', 'Image')
    b = cv2.getTrackbarPos('B', 'Image')
    img[:] = [b, g, r]

img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('Image')

# Trackbars für RGB-Werte hinzufügen
cv2.createTrackbar('R', 'Image', 0, 255, update_color)
cv2.createTrackbar('G', 'Image', 0, 255, update_color)
cv2.createTrackbar('B', 'Image', 0, 255, update_color)

while True:
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC zum Beenden
        break

cv2.destroyAllWindows()
