import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import requests
import pandas as pd
from datetime import datetime

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Real-Time Emotion AI", layout="wide")

# ----------------- CONSTANTS -----------------
EMOTION_GIFS = {
    "happy": "https://media.giphy.com/media/111ebonMs90YLu/giphy.gif",
    "sad": "https://media.giphy.com/media/OPU6wzx8JrHna/giphy.gif",
    "angry": "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",
    "fear": "https://media.giphy.com/media/3o6ZtaO9BZHcOjmErm/giphy.gif",
    "surprise": "https://media.giphy.com/media/l0Iyl55kTeh71nTXy/giphy.gif",
    "disgust": "https://media.giphy.com/media/3o7abB06u9bNzA8lu8/giphy.gif",
    "neutral": "https://media.giphy.com/media/3o6fJ1BM7r3Jkz2lI8/giphy.gif",
}

STATIC_CHEERUP = {
    "happy": "You look happy! Keep spreading that energy 😄",
    "sad": "It’s okay to feel sad sometimes 💛",
    "angry": "Take a deep breath. Peace matters ❤️",
    "fear": "Try slow breathing: inhale 4s, hold 2s, exhale 6s 💙",
    "surprise": "Life is full of surprises 😌",
    "disgust": "Not everything will be perfect — and that’s okay 🌱",
    "neutral": "A calm mind is a powerful tool 🙂",
}

PLAYLISTS = {
    "happy": "https://www.youtube.com/watch?v=3GwjfUFyY6M",
    "sad": "https://www.youtube.com/watch?v=lFcSrYw-ARY",
    "angry": "https://www.youtube.com/watch?v=2OEL4P1Rz04",
    "fear": "https://www.youtube.com/watch?v=MI2H0m-0mGE",
    "neutral": "https://www.youtube.com/watch?v=jfKfPfyJRdk",
    "surprise": "https://www.youtube.com/watch?v=QwZT7T-TXT0",
    "disgust": "https://www.youtube.com/watch?v=2OEL4P1Rz04",
}

# ----------------- HELPERS -----------------
def ai_cheer_up(emotion):
    prompt = f"The person looks {emotion}. Give a short caring, motivational message."

    API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"

    try:
        response = requests.post(API_URL, json={"inputs": prompt}, timeout=8)
        out = response.json()
        if isinstance(out, list) and "generated_text" in out[0]:
            return out[0]["generated_text"]
    except:
        pass

    return STATIC_CHEERUP.get(emotion, "Stay strong — everything will be okay 💛")


def draw_boxes(image_rgb, results):
    """Draw bounding boxes + emotion labels."""
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for r in results:
        region = r.get("region", {})
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        emo = r["dominant_emotion"].capitalize()

        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv2.putText(
            img_bgr,
            emo,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def analyze_emotions(rgb):
    result = DeepFace.analyze(
        img_path=rgb,
        actions=["emotion"],
        enforce_detection=False
    )
    if isinstance(result, list):
        return result
    return [result]


def log_history(source, emotion, details):
    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.session_state["history"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "emotion": emotion,
        "details": str(details)
    })

# ----------------- MAIN UI -----------------
st.title("😊 Real-Time Emotion Recognition with AI Mood Support")

MODE = st.radio("Choose Mode:", ["Image Mode", "Webcam Mode"])

if "history" not in st.session_state:
    st.session_state["history"] = []

# ----------------- IMAGE MODE -----------------
if MODE == "Image Mode":
    uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)

        col1, col2 = st.columns([1.3, 1])

        with col1:
            st.image(img_np, caption="Uploaded Image", width=450)

        if st.button("Analyze Emotion"):
            with st.spinner("Analyzing..."):
                results = analyze_emotions(img_np)

            main = results[0]
            emotion = main["dominant_emotion"]

            # Draw boxes (smaller)
            boxed = draw_boxes(img_np, results)

            with col1:
                st.image(boxed, caption="Detected Faces", width=450)

            with col2:
                st.subheader(f"Emotion: {emotion.capitalize()}")

                gif = EMOTION_GIFS.get(emotion)
                if gif:
                    st.image(gif, width=250)

                st.video(PLAYLISTS.get(emotion))

                st.subheader("💬 AI Support")
                st.info(ai_cheer_up(emotion))

            log_history("image", emotion, main["emotion"])

# ----------------- WEBCAM MODE (STABLE + BOXES) -----------------
elif MODE == "Webcam Mode":
    st.write("Capture a photo from webcam to detect emotions:")

    img = st.camera_input("Webcam", label_visibility="visible")

    if img is not None:

        col1, col2 = st.columns([1.3, 1])

        with col1:
            frame = Image.open(img).convert("RGB")
            rgb = np.array(frame)

            with st.spinner("Detecting emotions..."):
                results = analyze_emotions(rgb)

            main = results[0]
            emotion = main["dominant_emotion"]

            # Draw boxes (smaller image)
            boxed = draw_boxes(rgb, results)
            st.image(boxed, caption="Detected Faces", width=450)

        with col2:
            st.subheader(f"Emotion: {emotion.capitalize()}")

            # GIF (smaller)
            gif = EMOTION_GIFS.get(emotion)
            if gif:
                st.image(gif, width=250)

            # Music (smaller)
            playlist = PLAYLISTS.get(emotion)
            if playlist:
                st.video(playlist)

            # AI message
            st.subheader("💬 AI Support")
            st.info(ai_cheer_up(emotion))

            log_history("webcam", emotion, main["emotion"])


# ----------------- HISTORY -----------------
st.markdown("---")
st.subheader("📊 Emotion History")

if st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download History CSV",
        df.to_csv(index=False),
        file_name="emotion_history.csv",
        mime="text/csv"
    )
    
else:
    st.info("No history yet.")

st.markdown("---")
st.subheader("📉 Emotion Analytics Dashboard")

if st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])

    # ---------------- EMOTION FREQUENCY BAR CHART ----------------
    st.subheader("1️⃣ Emotion Frequency")

    freq = df["emotion"].value_counts()
    st.bar_chart(freq)

    # ---------------- EMOTION OVER-TIME TREND ----------------
    st.subheader("2️⃣ Emotion Trend Over Time")

    trend_df = df.copy()
    trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"])

    line_data = trend_df.groupby("timestamp")["emotion"].apply(lambda x: x.iloc[0])
    line_data = line_data.reset_index()

    # Convert emotions to numeric codes for line plot
    emotion_map = {e: i for i, e in enumerate(df["emotion"].unique())}
    line_data["emotion_code"] = line_data["emotion"].map(emotion_map)

    st.line_chart(line_data.set_index("timestamp")["emotion_code"])

    st.caption("Note: Emotions are encoded numerically for line graph visualization.")

    # ---------------- PROBABILITY RADAR CHART (optional) ----------------
    st.subheader("3️⃣ Latest Emotion Probability Distribution")

    last_entry = df.iloc[-1]
    try:
        import json
        probs = eval(last_entry["details"])  # stored dictionary string → dict
        
        prob_df = (
            pd.DataFrame(list(probs.items()), columns=["Emotion", "Probability"])
            .sort_values(by="Probability", ascending=False)
        )

        st.bar_chart(prob_df.set_index("Emotion"))

    except:
        st.info("Probability data unavailable for the last entry.")
else:
    st.info("No emotion analytics available yet.")

