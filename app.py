import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import requests

st.set_page_config(page_title="Real-Time Sentiment Analysis", layout="wide")

st.title("😊 Real-Time & Image-Based Sentiment Analysis App")
st.write("Analyze emotions using an uploaded image or in real-time using your webcam.")

# ---------------- AI Cheer-Up Function (FREE) ----------------
def ai_cheer_up(emotion):
    prompt = f"The person looks {emotion}. Give a short caring, positive, motivational message in one sentence."
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
    payload = {"inputs": prompt}

    try:
        response = requests.post(API_URL, json=payload, timeout=25)
        output = response.json()
        msg = output[0]["generated_text"]
        return msg

    except:
        return "Stay strong — everything will be okay 💛"


MODE = st.radio("Choose Mode:", ["Image Mode", "Real-Time Webcam"])

# ------------------- IMAGE MODE -------------------
# ------------------- IMAGE MODE -------------------
if MODE == "Image Mode":
    uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width=400)

        if st.button("Analyze Emotion"):
            with st.spinner("Analyzing..."):
                result = DeepFace.analyze(
                    img_path=np.array(img),
                    actions=['emotion'],
                    enforce_detection=False
                )

                if isinstance(result, list):
                    result = result[0]

                emotion = result['dominant_emotion']

                # NOW inside correct scope:
                st.success("Emotion detected!")
                st.subheader(f"Dominant Emotion: {emotion}")
                st.json(result["emotion"])

                st.subheader("💬 AI Support Message")
                st.info(ai_cheer_up(emotion))



# ------------------- REAL-TIME WEBCAM MODE -------------------
elif MODE == "Real-Time Webcam":

    st.write("Turn on the webcam to detect emotions live.")
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    # Placeholder for AI message OUTSIDE THE LOOP — this is the real fix
    MESSAGE_BOX = st.empty()

    camera = cv2.VideoCapture(0)

    # Session state variables
    if "last_emotion" not in st.session_state:
        st.session_state.last_emotion = None
    if "last_message" not in st.session_state:
        st.session_state.last_message = ""

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Unable to access webcam.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            result = DeepFace.analyze(
                rgb,
                actions=['emotion'],
                enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            emotion = result["dominant_emotion"]

            cv2.putText(
                rgb,
                emotion.capitalize(),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0),
                3
            )
        except:
            emotion = None

        # Update the live video feed
        FRAME_WINDOW.image(rgb)

        # Update message only when emotion changes
        if emotion and emotion != st.session_state.last_emotion:
            st.session_state.last_emotion = emotion
            st.session_state.last_message = ai_cheer_up(emotion)

            # Update the message box once
            MESSAGE_BOX.subheader(f"💬 AI Support Message for: {emotion.capitalize()}")
            MESSAGE_BOX.info(st.session_state.last_message)

    camera.release()
