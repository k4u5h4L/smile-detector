# smile Recognition

import cv2

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_data.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_data.detectMultiScale(roi_gray, 1.1, 22)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
        smiles = smile_data.detectMultiScale(roi_gray, 1.7, 37)
        
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
    return frame

if __name__ == '__main__':
    # Loading the cascades
    face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_data = cv2.CascadeClassifier('haarcascade_eye.xml')
    smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')

    # Doing some Face Recognition with the webcam
    video_capture = cv2.VideoCapture(0) # 0 is to use the inbuilt webcam, and 1 is to use an external webcam
    
    while True:
        _, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converting to gray scale
        canvas = cv2.flip(detect(gray, frame), 1) # mirroring the frame

        cv2.imshow('Video', canvas)
        
        # quit the capture when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
    video_capture.release()
    cv2.destroyAllWindows()
    
print("Program closed")
