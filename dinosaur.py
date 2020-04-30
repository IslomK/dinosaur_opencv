# organize imports
import cv2
import imutils
from pynput import keyboard
import numpy as np

kb = keyboard.Controller()
bg = None


def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    return (thresholded, cnts)


if __name__ == "__main__":

    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(2)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    # region of interest (ROI) coordinates
    top, right, bottom, left = 100, 0, 300, 700

    num_frames = 0

    while(True):
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        keypress = cv2.waitKey(1) & 0xFF

        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            obj = segment(gray)

            # check whether hand region is segmented
            if obj is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, cnts) = obj

                for c in cnts:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(thresholded, (x, y), (x + w, y + h), (36, 255, 12), 2)

                    if x <= 80:
                        cv2.putText(thresholded, "jump", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, 255)
                        kb.press(keyboard.Key.space)
                        kb.release(keyboard.Key.space)

                small = cv2.resize(thresholded, (0, 0), fx=0.5, fy=0.5)

                cv2.imshow("Thesholded", small)

        if keypress == ord("f"):
            num_frames = 31
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        small = cv2.resize(clone, (0, 0), fx=0.5, fy=0.5)

        cv2.namedWindow('ImageWindowName', cv2.WINDOW_NORMAL)
        cv2.imshow("ImageWindowName", small)
        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()