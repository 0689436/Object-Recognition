import cv2
import numpy as np
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers_indices = yolo_net.getUnconnectedOutLayers()
if isinstance(output_layers_indices, np.ndarray):
    output_layers_indices = output_layers_indices.flatten().tolist()
output_layers = [layer_names[i - 1] for i in output_layers_indices]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path
while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
