import cv2
import time
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture('Cricket Bowling 150fps.avi')
#print(int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi',fourcc, 20.0, (1280, 720))
while(cap.isOpened()):
    r, frame = cap.read()
    if r:
        start_time = time.time()
        frame = cv2.resize(frame, (1280, 720))  # Downscale to improve frame rate
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # HOG needs a grayscale image

        rects, weights = hog.detectMultiScale(gray_frame)

        # Measure elapsed time for detections
        end_time = time.time()
        print("Elapsed time:", end_time - start_time)

        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.35:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("preview", frame)
        out.write(frame)
    k = cv2.waitKey(10)
    if k & 0xFF == ord("q"):  # Exit condition
        break