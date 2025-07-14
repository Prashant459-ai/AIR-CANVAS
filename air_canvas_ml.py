import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Setup deque lists for different colors
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Indexes and colors
blue_index = green_index = red_index = yellow_index = 0
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Create a white canvas
paintWindow = np.full((471, 636, 3), 255, dtype=np.uint8)


def draw_buttons(frame):
    cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    cv2.rectangle(frame, (160, 1), (255, 65), colors[0], 2)
    cv2.rectangle(frame, (275, 1), (370, 65), colors[1], 2)
    cv2.rectangle(frame, (390, 1), (485, 65), colors[2], 2)
    cv2.rectangle(frame, (505, 1), (600, 65), colors[3], 2)
    cv2.putText(frame, "CLEAR", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, "BLUE", (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, "GREEN", (295, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, "RED", (420, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, "YELLOW", (515, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


save_counter = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    draw_buttons(frame)
    cv2.line(frame, (0, 65), (640, 65), (0, 0, 0), 2)
    cv2.putText(frame, "Draw below this line", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    result = hands.process(framergb)
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            landmarks = [(int(lm.x * 640), int(lm.y * 480)) for lm in handlms.landmark]
            mpDraw.draw_landmarks(frame, handlms, mpHands.HAND_CONNECTIONS)
            fore_finger = landmarks[8]
            middle_finger = landmarks[12]
            distance = np.linalg.norm(np.array(fore_finger) - np.array(middle_finger))

            if fore_finger[1] <= 65:
                if 40 <= fore_finger[0] <= 140:
                    bpoints = [deque(maxlen=1024)]
                    gpoints = [deque(maxlen=1024)]
                    rpoints = [deque(maxlen=1024)]
                    ypoints = [deque(maxlen=1024)]
                    blue_index = green_index = red_index = yellow_index = 0
                    paintWindow.fill(255)
                elif 160 <= fore_finger[0] <= 255:
                    colorIndex = 0
                elif 275 <= fore_finger[0] <= 370:
                    colorIndex = 1
                elif 390 <= fore_finger[0] <= 485:
                    colorIndex = 2
                elif 505 <= fore_finger[0] <= 600:
                    colorIndex = 3
            else:
                if distance > 40:
                    if colorIndex == 0:
                        bpoints.append(deque(maxlen=1024))
                        blue_index += 1
                    elif colorIndex == 1:
                        gpoints.append(deque(maxlen=1024))
                        green_index += 1
                    elif colorIndex == 2:
                        rpoints.append(deque(maxlen=1024))
                        red_index += 1
                    elif colorIndex == 3:
                        ypoints.append(deque(maxlen=1024))
                        yellow_index += 1
                else:
                    if fore_finger[1] > 65:
                        if colorIndex == 0:
                            bpoints[-1].appendleft(fore_finger)
                        elif colorIndex == 1:
                            gpoints[-1].appendleft(fore_finger)
                        elif colorIndex == 2:
                            rpoints[-1].appendleft(fore_finger)
                        elif colorIndex == 3:
                            ypoints[-1].appendleft(fore_finger)

    for i, points in enumerate([bpoints, gpoints, rpoints, ypoints]):
        for j in range(len(points)):
            for k in range(1, len(points[j])):
                if points[j][k - 1] and points[j][k]:
                    cv2.line(frame, points[j][k - 1], points[j][k], colors[i], 2)
                    cv2.line(paintWindow, points[j][k - 1], points[j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"drawing_{save_counter}.png"
        cv2.imwrite(filename, paintWindow)
        print(f"Drawing saved as {filename}")
        save_counter += 1

cap.release()
cv2.destroyAllWindows()