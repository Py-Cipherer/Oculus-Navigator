import cv2
import mediapipe as mp
import threading

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
running = True

def listen_stop():
    global running
    while running:
        user_input = input()
        if user_input.lower() == "stop":
            running = False

def get_finger_state(hand_landmarks):
    fingers=[]
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    tips=[8,12,16,20]
    joints=[6,10,14,18]

    for tip,joint in zip(tips,joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[joint].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def detect_gesture(fingers):
    if fingers == [0,0,0,0,0]:
        return "FIST / GHOOSA"
    elif fingers == [1,1,1,1,1]:
        return "PALM / THAPPAD"
    elif fingers == [1,0,0,0,0]:
        return "THUMBS UP / OKAY"
    elif fingers == [0,0,1,0,0]:
        return "F#CK YOU"
    elif fingers == [0,0,0,0,1]:
        return "EXCUSE ME!"
    elif fingers == [1,0,0,0,1]:
        return "CALL!"
    elif fingers == [1,1,0,0,1] or fingers == [0,1,0,0,1]:
        return "YO YO!!"
    else:
        return "UNKNOWN GESTURE"

threading.Thread(target=listen_stop, daemon=True).start()

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
    while running:
        success, img = cap.read()
        if not success:
            print("Camera not detected!")
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            print(results.multi_hand_landmarks)
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
                fingers = get_finger_state(hand)
                gesture = detect_gesture(fingers)
                cv2.putText(img, gesture, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

        cv2.imshow("Alex's Oculus Model 3", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

cap.release()
cv2.destroyAllWindows()
