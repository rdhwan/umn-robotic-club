import cv2
import numpy as np


# frame = cv2.imread("img/traffic_light.png")


def detect_green_color(frame: cv2.Mat) -> cv2.Mat:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_lower = np.array([140, 166, 58], np.uint8)
    red_upper = np.array([179, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)

    green_lower = np.array([36, 88, 83], np.uint8)
    green_upper = np.array([68, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    blue_lower = np.array([93, 60, 118], np.uint8)
    blue_upper = np.array([144, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)

    kernel = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

    # For green color
    green_mask = cv2.dilate(green_mask, kernel)
    res_green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(
                frame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255)
            )

    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(
        green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                "Green Colour",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
            )

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(
        blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.putText(
                frame,
                "Blue Colour",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
            )

    return frame


#
# # Program Termination
# while True:
# cv2.imshow("Sipaling robotic", frame)
# if cv2.waitKey(1) & 0xFF == ord("q"):
# cv2.destroyAllWindows()
# break

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    processed = detect_green_color(frame=frame)
    cv2.imshow("Capt", processed)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
