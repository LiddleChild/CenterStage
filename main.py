import cv2
import numpy as np
import pyvirtualcam

from util import *

CASCADE_SCALE = 1.3
INTERP_AMOUNT = 0.075
INTERP_THRESHOLD = 0.5

EYE_OFFSET = np.array((0, 10))

CROP_ASPECT_RATIO = np.array((4, 3))
OUTPUT_ASPECT_RATIO = np.array((16, 9))

CROP_SIZE = CROP_ASPECT_RATIO * 135
OUTPUT_SIZE = changeRatio(CROP_SIZE, OUTPUT_ASPECT_RATIO)

MAX_ZOOM_LEVEL = 40

REVERSE_VIRTUAL_CAM = False
REVERSE_SHOW_CAM = True

ENABLE_PREVIEW_CAM = True

ENABLE_ZOOM = True

global zoomLevel
zoomLevel = 0

def main():
    def onChange(val):
        global zoomLevel
        zoomLevel = val

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    SCREEN_SIZE = np.array(cap.read()[1].shape[:2][::-1], int)

    position = (SCREEN_SIZE - CROP_SIZE) // 2

    with pyvirtualcam.Camera(width=OUTPUT_SIZE[0], height=OUTPUT_SIZE[1], fps=60) as cam:
        print(f'Using virtual camera: {cam.device}')
        
        if (ENABLE_ZOOM):
            cv2.imshow("Window", np.zeros((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3)))
            cv2.createTrackbar("Zoom level", "Window", 0, MAX_ZOOM_LEVEL, onChange)

        while(True):
            _, img = cap.read()
            
            faces = face_cascade.detectMultiScale(img, CASCADE_SCALE, 5)
            eyes = eye_cascade.detectMultiScale(img, CASCADE_SCALE, 5)
            n = len(faces) + len(eyes)

            
            if (n > 0):
                mid = np.zeros((2), int)
                
                for i in eyes:
                    x, y, w, h = i
                    mid += np.array((x + w // 2, y + h // 2)) + EYE_OFFSET
                
                for i in faces:
                    x, y, w, h = i
                    mid += np.array((x + w // 2, y + h // 2))
                
                mid = mid // n

                position = interpolate(position, \
                    np.array(getTopLeftPos(mid, CROP_SIZE)), \
                    INTERP_AMOUNT, \
                    INTERP_THRESHOLD * (1 * zoomLevel / 100))

            # Crop to follow faces
            # img = cropImage(img, position, position + CROP_SIZE)

            # Crop to zoom
            zoomStart = np.array(position + CROP_SIZE * zoomLevel / 100, int)
            zoomEnd = np.array(position + CROP_SIZE - CROP_SIZE * zoomLevel / 100, int)
            img = cropImage(img, zoomStart, zoomEnd)

            # Resize to original size (fix zooming bug)
            img = cv2.resize(img, CROP_SIZE)
            img = changeImageRatio(img, OUTPUT_SIZE)

            # Show preview cam
            if (ENABLE_PREVIEW_CAM):
                cv2.imshow("Window", reverseCamera(img, REVERSE_SHOW_CAM))

            # Create virutal cam
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cam.send(reverseCamera(img, REVERSE_VIRTUAL_CAM))
            cam.sleep_until_next_frame()
            
            if (cv2.waitKey(1) & 0xFF == ord("q")):
                break

        cap.release()

        cv2.destroyAllWindows()

if (__name__ == "__main__"): main()
