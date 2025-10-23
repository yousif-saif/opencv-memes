import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import math
import numpy as np

detector = FaceMeshDetector()
cap = cv2.VideoCapture(0)


pig = cv2.imread("./dog.png", cv2.IMREAD_UNCHANGED)
pig = cv2.resize(pig, (200, 200))
# pig = cv2.cvtColor(pig, cv2.COLOR_BGR2RGB)

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (800, 600))
    if ret:
        img = cv2.flip(img, 1)
        
        face, bbox = detector.findFaceMesh(img, draw=False)
        
        if bbox:    
            first_face = bbox[0]
            top = first_face[9]
            bottom = first_face[164]
            left = first_face[123]
            right = first_face[358]
            
            # height = int(math.dist(top, bottom))
            # width = int(math.dist(left, right))
            height, width = pig.shape[:2]
            
            if top[1] + height > face.shape[0] or top[0] + width > face.shape[1]:
                continue  # skip if overlay is out of bounds
        
            overlay_rgb = pig[:, :, :3]
            overlay_alpha = pig[:, :, 3:] / 255.0
            
            roi = face[top[1]:top[1]+height, left[0]:left[0]+width]
            
            blended = (overlay_alpha * overlay_rgb + (1 - overlay_alpha) * roi).astype(np.uint8)

            face[top[1]:top[1]+height, left[0]:left[0]+width] = blended

            
            # cv2.circle(face, top, 5, (255, 0, 0), cv2.FILLED)
            # cv2.circle(face, bottom, 5, (255, 0, 0), cv2.FILLED)
            # cv2.circle(face, left, 5, (255, 0, 0), cv2.FILLED)
            # cv2.circle(face, right, 5, (255, 0, 0), cv2.FILLED)
            
            
            
            cv2.imshow("filter", face)
            
        cv2.imshow("filter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


