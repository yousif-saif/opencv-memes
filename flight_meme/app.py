import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import math
import os


detector = FaceMeshDetector(maxFaces=1)
cap = cv2.VideoCapture(0)

dirname = os.getcwd()

front_flight = cv2.imread(os.path.join(dirname, "flight_meme", "front.png"))
side_flight = cv2.imread(os.path.join(dirname, "flight_meme", "side.png"))

front_flight = cv2.resize(front_flight, (244, 300))
side_flight = cv2.resize(side_flight, (244, 300))


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    
    img_h, img_w, _ = front_flight.shape
    frame_h, frame_w, _ = frame.shape
    
    try:
        face, bbox = detector.findFaceMesh(frame)
        bbox = bbox[0]
        
        face_right_side_point = bbox[137]
        face_left_side_point = bbox[366]
        
        cv2.circle(face, face_right_side_point, 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(face, face_left_side_point, 5, (255, 0, 0), cv2.FILLED)
        
        
        # print(math.dist(face_left_side_point, face_right_side_point))
        if math.dist(face_left_side_point, face_right_side_point) < 150:
            # print("side flight")
            frame[0:img_h, frame_w-img_w:frame_w] = side_flight
            
        
        else:
            # print("front flight")
            frame[0:img_h, frame_w-img_w:frame_w] = front_flight
            
        
            
        
        cv2.imshow("flight meme", face)

    except Exception as e:
        cv2.imshow("flight meme", frame)
        print(e)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    

cv2.destroyAllWindows()
cap.release()