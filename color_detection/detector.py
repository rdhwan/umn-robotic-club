"""Color finder with opencv"""

import cv2
import numpy as np


def detect_rgb(
    frame: cv2.Mat,
    red_low: list[int],
    red_hi: list[int],
    green_low: list[int],
    green_hi: list[int],
    blue_low: list[int],
    blue_hi: list[int],
) -> cv2.Mat:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(
        hsv_frame, np.array(red_low, np.uint8), np.array(red_hi, np.uint8)
    )
    green_mask = cv2.inRange(
        hsv_frame, np.array(green_low, np.uint8), np.array(green_hi, np.uint8)
    )
    blue_mask = cv2.inRange(
        hsv_frame, np.array(blue_low, np.uint8), np.array(blue_hi, np.uint8)
    )

    kernel = np.ones((5, 5), "uint8")

    # Dilating
    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

    # NOTE: This remains unused.
    # and bitwise
    # res_red = cv2.bitwise_and(frame, frame, mask=red_mask)
    # res_green = cv2.bitwise_and(frame, frame, mask=green_mask)
    # res_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
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

    for contour in contours:
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

    for contour in contours:
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
