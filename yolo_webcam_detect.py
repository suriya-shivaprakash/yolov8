from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)  # Change 0 to 1 or 2 if webcam doesn't open

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()     
    
    if not ret:
        print("❌ Can't receive frame (stream end?). Exiting ...")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

