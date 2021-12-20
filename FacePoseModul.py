import cv2
import mediapipe as mp
import time
import numpy as np

class FacePoseDetector():

    def __init__(self, staticMode=False, maxFaces=1, SmoothLandmarks=True, EnbSeg=False, SmoothSeg=True, refLandmarks=False,
                 minDetectionCon=0.5, minTrackCon=0.5):

        self.maxFaces = maxFaces
        self.SmoothLandmarks = SmoothLandmarks
        self.EnbSeg = EnbSeg
        self.SmoothSeg = SmoothSeg
        self.refLandmarks = refLandmarks
        self.staticMode = staticMode
        self.minTrackCon = minTrackCon
        self.minDetectionCon = minDetectionCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_holistic = mp.solutions.holistic

        self.drawing_spec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.holistic = self.mp_holistic.Holistic(self.staticMode, self.maxFaces, self.SmoothLandmarks, self.EnbSeg, self.SmoothSeg,
                                                  self.staticMode, self.minDetectionCon, self.minTrackCon)


    def findFacePose(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(imageRGB)
        if self.results.face_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(image, self.results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                                                   connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
            # Netz Contour
                #self.mpDraw.draw_landmarks(image, self.results.face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                                              #landmark_drawing_spec=None, connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                self.mpDraw.draw_landmarks(image, self.results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                               landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        return image

    def findFacePosition(self, image, draw=True):
        self.lmList = []
        if self.results.holistic.process:
            for id, lm in enumerate(self.results.holistic_landmarks):
                #print(lm)#Posotion landmark x,y and z
                ih, iw, ic = image.shape # get shape of original frame
                cx, cy = int(lm.x * iw), int(lm.y * ih), int(lm.z * ic)
                self.lmList.append([id, cx, cy])
                if draw:
                #ID Face Detection
                    cv2.putText(image, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)

                    #print(id,x,y)

        return image


def main():

    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FacePoseDetector()
    while True:
        success, image = cap.read()
        image = detector.findFacePose(image) # if I need only ID Nommer whrite image, face = detector.findFaceMesh(image, False)
        lmList = detector.findFacePose(image, draw=False)
        if len(lmList) != 0:
            print(lmList[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow('Image', image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()