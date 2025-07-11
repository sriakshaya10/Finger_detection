import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

#  landmark indices
finger_tips = [4, 8, 12, 16, 20]

#  find hands & draw landmarks
def find_hands(img, draw=True):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            if draw:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return img, res

#  landmark positions
def find_position(img, res, hand_no=0, draw=True):
    lm_list = []
    if res.multi_hand_landmarks:
        my_hand = res.multi_hand_landmarks[hand_no]
        for id, lm in enumerate(my_hand.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
    return lm_list

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    success, frame = video_capture.read()
    if not success:
        break

    frame, res = find_hands(frame)
    landmarks = find_position(frame, res, draw=False)
    if landmarks:
        fingers_status = []
        # Thumb: compare x-coordinates(open/close)
        if landmarks[finger_tips[0]][1] > landmarks[finger_tips[0] - 1][1]:
            fingers_status.append(1)
        else:
            fingers_status.append(0)
        # Other fingers: compare y-coordinates(open/close)
        for finger in range(1, 5):
            if landmarks[finger_tips[finger]][2] < landmarks[finger_tips[finger] - 2][2]:
                fingers_status.append(1)
            else:
                fingers_status.append(0)
        total_fingers_up = fingers_status.count(1)
        # Display the count
        cv2.rectangle(frame, (10, 10), (100, 70), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, str(total_fingers_up), (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 4)

    cv2.imshow("Finger Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
