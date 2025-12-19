import cv2
import mediapipe as mp
#import threading
import numpy as np

mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils

cap= cv2.VideoCapture(0)
running = True

"""
def listen_stop():
    global running
    while running:
        user_input=input()
        if user_input.lower() == "stop" or user_input.lower() == "exit" or user_input.lower() == "quit":
            runnning = False

threading.Thread(target = listen_stop , daemon=True).start()
"""

def fin_deg(p1,p2,p3):
    
    p1 = np.array([p1.x, p1.y, p1.z])
    p2 = np.array([p2.x, p2.y, p2.z])
    p3 = np.array([p3.x, p3.y, p3.z])

    t1= p1-p2
    t2= p3-p2

    t1_u = t1/np.linalg.norm(t1)
    t2_u = t2/np.linalg.norm(t2)

    dot_t= np.clip(np.dot(t1_u, t2_u), -1.0, 1.0)
    t_rad= np.arccos(dot_t)
    t_deg= np.degrees(t_rad)

    return t_deg



def finger_up(fin,mcp,pip,dip,tip):
    #bhooliyo mat, yaad krle naam
    #mcp= metacarpophalanges
    #pip= proximal-interphalanges
    #dip= diatal-interphalanges
    #tip= finger tip

    pip_angle= fin_deg(fin.landmark[mcp],fin.landmark[pip],fin.landmark[dip])
    #tip_angle= fin_deg(fin.landmark[pip],fin.landmark[dip],fin.landmark[tip])
    #print(f"PIP Angle: {pip_angle:.2f}, TIP Angle: {tip_angle:.2f}")
    if pip_angle>140: #and tip_angle>165:
        return 1
    
    return 0



def detect_gesture(fingers):
    s=sum(fingers)
    if fingers == [0,0,0,0,0]:
        return "FIST / ZERO-0"
    elif fingers == [1,1,1,1,1]:
        return "PALM / FIVE-5"
    elif fingers == [0,1,1,0,0]:
        return "PEACE / TWO-2"
    elif fingers == [1,0,0,0,0]:
        return "THUMBS UP / OKAY "
    elif fingers == [0,0,0,0,1]:
        return "EXCUSE ME!"
    elif fingers == [1,0,0,0,1]:
        return "CALL!"
    elif fingers == [1,1,0,0,0]:
        return "DIRECT / SHOOT"
    elif fingers == [1,1,0,0,1] or fingers == [0,1,0,0,1]:
        return "SPIDER-MAN / YO!"
    elif fingers == [0,1,0,0,0]:
        return "ONE-1"
    elif fingers == [0,1,1,1,0]:
        return "THREE-3"
    elif fingers == [0,1,1,1,1]:
        return "FOUR-4"
    else:
        return f"UNKNOWN GESTURE | {s} FINGERS UP"



print("\n----------------------------------------------------------------\nModel Started, Press 'q' to stop!\n----------------------------------------------------------------\n")

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while running:
        success, img = cap.read()
        if not success:
            print("Camera not detected, jaake theek kar!")
            break

        img = cv2.flip(img,1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                world_landmarks = result.multi_hand_world_landmarks[i]

                fingers = [finger_up(world_landmarks,2,3,4,1),
                           finger_up(world_landmarks,5,6,7,8),
                           finger_up(world_landmarks,9,10,11,12),
                           finger_up(world_landmarks,13,14,15,16),
                           finger_up(world_landmarks,17,18,19,20)]

                gesture = detect_gesture(fingers)
                cv2.putText(img, gesture, (50, 80 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (8,143,143), 2)

                mp_draw.draw_landmarks(img, hand_landmarks,
                                       mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0,0,255),thickness=2, circle_radius=1),
                                       mp_draw.DrawingSpec(color=(0,255,0), thickness=2))

        cv2.imshow("Alex's Oculus Model 4", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

cap.release()
cv2.destroyAllWindows()

        
"""
        if result.multi_hand_landmarks:
            for hand_landmarks,handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

                if result.multi_hand_world_landmarks:
            
                    for world_landmarks in result.multi_hand_world_landmarks:
                        #fingers = [thumb,index,middle,ring,little]
                        fingers = [finger_up(world_landmarks,1,2,3,4),
                                   finger_up(world_landmarks,5,6,7,8),
                                   finger_up(world_landmarks,9,10,11,12),
                                   finger_up(world_landmarks,13,14,15,16),
                                   finger_up(world_landmarks,17,18,19,20)]
                        gesture= detect_gesture(fingers)
                        cv2.putText(img, gesture, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (8,143,143),2)
                        
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0,0,255),thickness=2, circle_radius=1 ),
                                       mp_draw.DrawingSpec(color=(0,255,0), thickness=2))
"""
