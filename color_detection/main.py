import cv2
import detector
import time

# TODO: EDIT VALUE CONFIG HERE
# NOTE: low-high hsv value depends on what camera you're using
RED_LOW = [140, 166, 58]
RED_HIGH = [179, 255, 255]

GREEN_LOW = [36, 88, 83]
GREEN_HIGH = [68, 255, 255]

BLUE_LOW = [95, 66, 109]
BLUE_HIGH = [130, 255, 255]


# detect color on single frame
frame = cv2.imread("img/sample_blue.jpg")

start = time.perf_counter()
processed = detector.detect_rgb(
    frame=frame,
    red_low=RED_LOW,
    red_hi=RED_HIGH,
    green_low=GREEN_LOW,
    green_hi=GREEN_HIGH,
    blue_low=BLUE_LOW,
    blue_hi=BLUE_HIGH,
)
end = time.perf_counter()

print(f"Process took: {end - start}")

while True:
    cv2.imshow("sample", processed)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# # using VideoCapture to get video feeds
# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()

#     processed = detector.detect_rgb(frame=frame)
#     cv2.imshow("Capt", processed)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()

cv2.destroyAllWindows()
