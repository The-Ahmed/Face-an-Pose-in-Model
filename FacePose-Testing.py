import cv2
import time
import FacePoseModul as fpm

cap = cv2.VideoCapture(0)
pTime = 0
detector = fpm.FacePoseDetector()
while True:
    success, image = cap.read()
    image = detector.findFacePose(image)  # if I need only ID Nommer whrite image, face = detector.findFaceMesh(image, False)
    lmList = detector.findFacePose(image, draw=False)
    if len(lmList) != 0:
        print(lmList[0])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Image', image)
    cv2.waitKey(1)