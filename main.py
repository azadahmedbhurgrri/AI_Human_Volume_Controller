import cv2
import math
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

st.title("AI Human Volume Controller")
st.write("Apne hath ki Shahadat ki ungli (Index Finger) aur Angoothe (Thumb) ke darmiyan distance badha kar ya kam kar ke volume control karein.")

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.vol_per = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Image ko flip karna horizontal mirroring ke liye
        img = cv2.flip(img, 1)
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        lmList = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    
        if len(lmList) != 0:
            # Thumb aur Index finger tips
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
            
            # Distance calculate karna
            length = math.hypot(x2 - x1, y2 - y1)
            
            # Distance ko 0-100% mein convert karna
            self.vol_per = int(np.interp(length, [20, 180], [0, 100]))
            
            if length < 20:
                cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
                
            # Screen par Volume percentage likhna
            cv2.putText(img, f'Volume: {self.vol_per}%', (40, 50), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        return frame.from_ndarray(img, format="bgr24")

# Streamlit WebRTC Component jo browser ka camera use karega
webrtc_streamer(
    key="volume-control",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
