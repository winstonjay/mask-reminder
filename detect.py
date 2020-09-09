import cv2
import os

cv_path = os.path.expanduser("~") + '/opencv/data/haarcascades/'
faceCascade = cv2.CascadeClassifier(cv_path + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)


def extract_image(img, points):
    (x, y, w, h) = points
    crop = img[y:y+h, x:x+w]
    H, W, *_ = img.shape
    return cv2.resize(crop, (W, H), interpolation = cv2.INTER_AREA)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if len(faces) > 0:
        frame = extract_image(frame, faces[0])
        cv2.putText(frame,"Where's your Mask?", (100,500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
        cv2.imshow('Video', frame)
    else:
        # Display the resulting frame
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()