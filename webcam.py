import cv2 as cv
import numpy as np

# url ="http://10.99.246.157:8080/shot.jpg"
count = 1
while(True):

    url ="http://10.99.246.157:8080/shot.jpg"           

    cp=cv.VideoCapture(url)
    while(True):
        ret, frame=cp.read()
        if frame is not None:
            cv.namedWindow("frame",0)
            cv.imshow("frame",frame)
        q=cv.waitKey(0)
        if q==ord("q"):
            break
        elif q==ord("s"):         
            print(777)
            cv.imwrite('test' + str(count) + '.png', frame)
            count = count + 1
            break
    cv.destroyAllWindows()
    
# import numpy as np
# import cv2

# cap = cv2.VideoCapture('http://100.115.118.225:8080/shot.jpg')

# while(True):
#     ret, frame = cap.read()

#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()