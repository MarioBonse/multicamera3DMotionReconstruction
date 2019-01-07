
import cv2
from operator import xor


def callback(value):
    pass


def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)



def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values

def findHSVparameter(name):
    image = cv2.imread(name)
    rgb = image
    blurred = cv2.GaussianBlur(rgb,(5,5),0)
    frame_to_thresh = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    setup_trackbars("HSV")
    while True:
        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values("HSV")

        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cv2.imshow("Original", image)
        cv2.imshow("Thresh", thresh)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            break

findHSVparameter("img0.jpg")
