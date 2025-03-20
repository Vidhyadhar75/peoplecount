import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8n.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(r"C:\Users\HP\OneDrive\Desktop\cc cameras\New folder (3)\CSE NVr-2_CSE Entrance_20250319085917_20250319091959.mp4")

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0
tracker = Tracker()

# Define the line (x1, y1) to (x2, y2)
line = [(128,332), (1005,257)]
#line = [(196,167), (614,360)]
crossed_in = set()  # IDs of people who crossed into the area
crossed_out = set()  # IDs of people who crossed out of the area

# Function to determine the position of a point relative to a line
def point_position(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

# Function to check if two points are on opposite sides of a line
def is_crossing_line(prev_pos, curr_pos, line):
    (a, b) = line
    prev_side = point_position(a, b, prev_pos)
    curr_side = point_position(a, b, curr_pos)
    return prev_side * curr_side < 0  # Opposite sides

# Function to determine the direction of crossing
def crossing_direction(prev_pos, curr_pos, line):
    (a, b) = line
    prev_side = point_position(a, b, prev_pos)
    curr_side = point_position(a, b, curr_pos)
    if prev_side < 0 and curr_side >= 0:  # Crossing from left to right (in)
        return "in"
    elif prev_side >= 0 and curr_side < 0:  # Crossing from right to left (out)
        return "out"
    return None

# Dictionary to store previous positions of each ID
prev_positions = {}

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data

    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if c == 'person':  # Only consider persons
            list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        # Get the previous position of the object
        prev_pos = prev_positions.get(id, (cx, cy))

        # Check if the object crossed the line
        if is_crossing_line(prev_pos, (cx, cy), line):
            direction = crossing_direction(prev_pos, (cx, cy), line)
            if direction == "in":
                crossed_in.add(id)
            elif direction == "out":
                crossed_out.add(id)

        # Update the previous position
        prev_positions[id] = (cx, cy)

        # Draw bounding box and ID
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
        cv2.putText(frame, str(int(id)), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    # Draw the line
    cv2.line(frame, line[0], line[1], (0, 0, 255), 2)

    # Display the counts of persons who crossed in and out
    cv2.putText(frame, f"In: {len(crossed_in)}", (50, 80), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(frame, f"Out: {len(crossed_out)}", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
