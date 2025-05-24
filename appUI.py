import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
import math
import time
import collections

# ====================== SETUP ======================
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .css-18e3th9 {
            padding-top: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .block-container {
            padding-top: 1rem;
        }
        .stButton>button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
        .stTextArea textarea {
            font-size: 1.2rem;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('sign_language_alphaDigi_model.h5')

model = load_model()

# Constants
imgSize = 128
offset = 20
labels = list("0123456789abcdefghijklmnopqrstuvwxyz")
prediction_buffer = collections.deque(maxlen=8)

# State Initialization
if 'camera' not in st.session_state:
    st.session_state.camera = False
if 'text_output' not in st.session_state:
    st.session_state.text_output = ""
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'current_char_buffer' not in st.session_state:
    st.session_state.current_char_buffer = []
if 'char_start_time' not in st.session_state:
    st.session_state.char_start_time = None
if 'prev_hand_detected' not in st.session_state:
    st.session_state.prev_hand_detected = False
if 'unique_key' not in st.session_state:
    st.session_state.unique_key = 0

# ====================== UI LAYOUT ======================
st.markdown("<h1 style='text-align: left;'>Indian Sign Language Detection</h1>", unsafe_allow_html=True)
st.caption("Show signs (digits + alphabets) to the camera to build words.")

col_camera, col_controls = st.columns([7, 3])

with col_camera:
    camera_placeholder = st.empty()

with col_controls:
    st.markdown("### Controls")
    if st.button("Open Camera"):
        st.session_state.camera = True
    if st.button("Close Camera"):
        st.session_state.camera = False
        st.session_state.text_output = ""
        st.session_state.unique_key = 0
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
    if st.button("Clear Text"):
        st.session_state.text_output = ""
    st.markdown("### Recognized Text")
    text_box = st.empty()

# ====================== PROCESSING ======================
def preprocess_image(img):
    img = cv2.resize(img, (imgSize, imgSize))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img_final = np.expand_dims(img, axis=0)
    return img_final, img

if st.session_state.camera and st.session_state.cap is None:
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.detector = HandDetector(maxHands=2, detectionCon=0.8, minTrackCon=0.5)

if st.session_state.camera and st.session_state.cap is not None:
    cap = st.session_state.cap
    detector = st.session_state.detector

    while st.session_state.camera:
        success, img = cap.read()
        if not success:
            st.error("Failed to capture frame.")
            st.session_state.camera = False
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            x_min = min(hand['bbox'][0] for hand in hands) - offset
            y_min = min(hand['bbox'][1] for hand in hands) - offset
            x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands) + offset
            y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands) + offset
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(img.shape[1], x_max), min(img.shape[0], y_max)

            imgCrop = img[y_min:y_max, x_min:x_max]
            aspectRatio = (y_max - y_min) / (x_max - x_min)
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            if aspectRatio > 1:
                k = imgSize / (y_max - y_min)
                wCal = math.ceil(k * (x_max - x_min))
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / (x_max - x_min)
                hCal = math.ceil(k * (y_max - y_min))
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            processed_img, normalized_img = preprocess_image(imgWhite)
            prediction = model.predict(processed_img, verbose=0)
            index = np.argmax(prediction[0])
            confidence = prediction[0][index]
            predicted_char = labels[index]

            prediction_buffer.append(predicted_char)
            smoothed_pred = max(set(prediction_buffer), key=prediction_buffer.count)

            st.session_state.current_char_buffer.append(predicted_char)
            st.session_state.prev_hand_detected = True

            if st.session_state.char_start_time is None:
                st.session_state.char_start_time = time.time()

            cv2.rectangle(imgOutput, (x_min, y_min - 70), (x_min + 250, y_min - 20), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, f"{smoothed_pred} ({confidence:.2f})", (x_min, y_min - 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x_min, y_min), (x_max, y_max), (255, 0, 255), 4)

        else:
            if st.session_state.prev_hand_detected and st.session_state.current_char_buffer:
                # Decide final letter shown the longest
                counter = collections.Counter(st.session_state.current_char_buffer)
                final_char = counter.most_common(1)[0][0]
                st.session_state.text_output += final_char
                st.session_state.prev_hand_detected = False
                st.session_state.char_start_time = None
                st.session_state.current_char_buffer = []
                st.session_state.unique_key += 1
                text_box.text_area("Recognized Text", value=st.session_state.text_output, height=100, max_chars=1000, key=st.session_state.unique_key)

        # Display camera
        resized_output = cv2.resize(imgOutput, (800, 600))
        camera_placeholder.image(cv2.cvtColor(resized_output, cv2.COLOR_BGR2RGB), channels="RGB")

        time.sleep(0.001)  # 1ms sleep instead of 10ms

    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None