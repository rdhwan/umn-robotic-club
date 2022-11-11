import cv2

cap = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while True:
    _, frame = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow(str(fps), frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
