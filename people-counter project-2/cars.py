import cv2
import cvzone
from ultralytics import YOLO
import math
from sort import *
cap=cv2.VideoCapture("../videos/people.mp4")

model = YOLO("../Yolo-weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
mask=cv2.imread("mask.png")

#Tracking
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3) #here tracker is a object that holds mentioned properties

limits=[400,297,673,297]
totalCount=[]

while True:
    success, img=cap.read()
    imgRegion=cv2.bitwise_and(img,mask)

    imgGraphics=cv2.imread("img.png", cv2.IMREAD_UNCHANGED)
    img=cvzone.overlayPNG(img,imgGraphics,(0,0))
    results= model(imgRegion, stream=True)

    detections=np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Bounding Box
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w,h=x2-x1,y2-y1


            #Confidence
            conf=math.ceil((box.conf[0]*100))/100
            print(conf)


            #class name
            cls=int(box.cls[0])
            if conf>0.5 :
                # cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.6,thickness=1) #the format is (image,string to display on boxes,(x,y))
             #so in cls=int(box.cls[0]) it gives index number based on image, like 0 for person and etc. That values are predefined while training the model.
            # and I used the same object list as classNames above so that I can access the names of the object by indexing i.e classNames[cls]. (where cls is index)
            #     cvzone.cornerRect(img, (x1, y1, w, h), l=10)

                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))

# working: detections  →  tracker.update()  →  resultsTracker
        # (x1,y1,x2,y2,conf)              (x1,y1,x2,y2,ID)

        resultsTracker=tracker.update(detections)

        # cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5) #it adds the line
        for result in resultsTracker:
            x1,y1,x2,y2,id=result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=2,colorR=(255,0,255))
            cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,offset=10)

            cx =  x1 + w//2
            cy = y1 + h // 2
            cv2.circle(img, (int(cx), int(cy)), 5, (255,0,255),cv2.FILLED)
            if limits[0]< cx <limits[2] and limits[1]-15 < cy < limits[1]+15:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # cvzone.putTextRect(img,f'Count: {len(totalCount)}', (50,50))
        print(x1, y1, x2, y2)

    # cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_SIMPLEX,3,(50,50,255),8)

    cv2.imshow("Image",img)
    # cv2.imshow("ImageRegion",imgRegion)
    cv2.waitKey(1)

