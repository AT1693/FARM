
import cv2
import numpy as np
import serial
import time

cap = cv2.VideoCapture(1)
capx = cv2.VideoCapture(2)

ser=serial.Serial('COM16', 9600, timeout=1)

line2 = 320

#Processing the vedio frame by frame
while(cap.isOpened()):
    #Reading the frame. ret returs true or false. If its false than vedio has finished.
    ret, orj_frame = cap.read()
    retx, orj_framex = capx.read()

    #To run the vedio for infinte time
    '''if not ret:
        #Loading the vedio again
        vedio = cv2.VideoCapture(videoPath)
        continue'''

    if ret==False or retx==False:
        continue

    #To run the vedio for infinte time
    '''if not retx:
        #Loading the vedio again
        vediox = cv2.VideoCapture(videoPathx)
        continue'''
    
    #Crop image to extract our focus area
    frame = orj_frame[280:480, :640]

    
    #Processing image for higher accuracy
    frame = cv2.GaussianBlur(frame, (5,5), 6)
    #frame = cv2.medianBlur(frame,7)
    
    #Converting RGB colorspace to HSV(Hue(H), Saturation(S) and Value(V)) colorspace for extracting a colored object
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Creating a range of color that need to be detected
    low_green = np.array([50,30,30])
    up_green = np.array([150,255,255])

    #Selection only those colors that fall in the specified range
    mask = cv2.inRange(hsv, low_green, up_green)

    kernel = np.ones((28,28),'int')
    dilated = cv2.dilate(mask,kernel)

    b = dilated.T

    #Processing image for higher accuracy
    framex = cv2.GaussianBlur(orj_framex, (5,5), 6)
    #frame = cv2.medianBlur(frame,7)
    
    #Converting RGB colorspace to HSV(Hue(H), Saturation(S) and Value(V)) colorspace for extracting a colored object
    hsvx = cv2.cvtColor(framex, cv2.COLOR_BGR2HSV)

    #Creating a range of color that need to be detected
    low_wh = np.array([25,50,50])
    up_wh = np.array([32,255,255])

    #Selection only those colors that fall in the specified range
    maskx = cv2.inRange(hsvx, low_wh, up_wh)

    kernelx = np.ones((5,5),'int')
    dilatedx = cv2.dilate(maskx,kernelx)

    if(np.count_nonzero(dilatedx[-1:-6][:] == 255)>0):
        sx = b"5"
    else:
        sx = b"4"

    data = []
    for i in range (b.shape[0]):
        t = np.array(b[i][:])
        cnt = np.count_nonzero(t == 255)
        for j in range(cnt):
            data.append(i)
    try:
        #line = int(np.mean(data))
        line = int(np.median(data))
    except Exception:
        s = b'0'
        ser.write(s)
        continue
    else:

        fline = np.zeros((mask.shape), np.uint8)
        fline[:,(-mask.shape[1]//6+mask.shape[1]//2)] = (124)
        fline[:,(mask.shape[1]//6+mask.shape[1]//2)] = (124)
        fline[:,line] = (255)

        if(line<(-mask.shape[1]//6+mask.shape[1]//2)):
            s = b'1'
        elif (line>(mask.shape[1]//6+mask.shape[1]//2)):
            s = b'3'
        else:
            s = b'2'

        #for i in range(50):
        ser.write(s)
        ser.write(sx)
        print(s)
        print(sx)
        #msg = ser.readline()
        #print ("Message from arduino: ")
        #print (msg)
        
        cv2.imshow("frame",frame)
        cv2.imshow("edges",dilated)
        cv2.imshow("turn",fline)

        cv2.imshow("framex",framex)
        cv2.imshow("edgesx",dilatedx)

        time.sleep(0.5)
        ser.close()
        key=cv2.waitKey(25)
        if(key==27):
            break
#Calling the Destructor
#ser.close()
vedio.release()
vediox.release()
cv2.destroyAllWindows()
