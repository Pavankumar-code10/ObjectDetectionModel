# from ultralytics import YOLO
# import cv2
# model =YOLO('../Yolo-weights/yolov8l.pt')
# results = model("Images/images2.jpg", imgsz=(640,640),show=True)  # (height, width)
# cv2.waitKey(0)
##THE PROBLEM WAS THE DISPLAY POP WINDOW WAS NOT STAYING SO I USED---->


from ultralytics import YOLO
import cv2
model = YOLO('../Yolo-weights/yolov8l.pt')
results = model("Images/images.jpg", imgsz=(1280,720))
annotated = results[0].plot()
cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
##THE DISPLAY WINDOW STAYS



# from ultralytics import YOLO #ultalytics is library
# import cv2 #OpenCV a image processing library
# import matplotlib.pyplot as plt #its basically that new window(plot) that opens to display image
#
# model = YOLO('yolov8n.pt') #here Iam loading a pre-trained model by importing a file(yolov8n.pt). This file has already know how to detect 80 common objects like care, bag, dog, etc
# results = model("Images/images.jpg") #here Iam inserting the image into the model
#
# annotated = results[0].plot(line_width=1, font_size=0.4) #draws the box boundary around the objects.
#
# plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))#cv2.cvtColor swaps the color channels so your image doesn't look weirdly colored when displayed.
# plt.axis('off')#turns of the x and y axis points.
# plt.title("YOLOv8 Detection")# gives a title to plot display, its written on the top
# plt.show()#finally displays the plot

##THE ABOVE METHOD, CREATES IT'S OWN PLOT WINDOW TO DISPLAY THE IMAGE, DOESN'T GIVE A SHI*T ABOUT POP WINDOW