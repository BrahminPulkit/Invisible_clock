import cv2
import numpy as np

cap = cv2.VideoCapture(0)
back = cv2.imread('./image.jpg')

while cap.isOpened():
    #take each frame
    ret, frame = cap.read()

    if ret:
        # how we convert rgb to hsv?
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv",hsv)
        #how we get hsv value?
        #lower: hue - 10,100,100,  heigher : h+10,255,255
        red = np.uint8([[[0,0,255]]]) #bgr value of red
        hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
        #get the hsv value of red from hsv
        #print(hsv_red)

        #threshold the hsv value to get only red color
        l_red = np.array([0,100,100])
        u_red = np.array([0,255,255])
        
        mask = cv2.inRange(hsv, l_red, u_red)
        #cv2.imshow("mask",mask)

        #all things red
        part_1 = cv2.bitwise_and(back , back , mask=mask)
        #cv2.imshow("part_1", part_1)

        mask = cv2.bitwise_not(mask)

        #part_2 all things is not black
        part_2 = cv2.bitwise_and(frame, frame, mask= mask)
        #cv2.imshow("part_2", part_2)

        cv2.imshow("cloak", part_1 + part_2)

        if cv2.waitKey(5) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
