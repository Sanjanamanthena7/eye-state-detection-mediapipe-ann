# ==========================================================
# EYE STATE DETECTION
# MediaPipe + OpenCV + ANN + Streamlit
# ==========================================================

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# IMPORT PROJECT MODULES
# ==========================================================

from utils.mediapipe_utils import (
    create_landmarker,
    extract_eye_features
)

from utils.predictor import (
    predict_eye_state
)


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Eye State Detection",
    page_icon="👁️",
    layout="wide"
)


# ==========================================================
# CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fff8;
    }

    .title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        color: #228B22;
    }

    .subtitle {
        text-align: center;
        color: gray;
        font-size: 18px;
    }

    .footer {
        text-align: center;
        color: gray;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    "<div class='title'>👁️ Eye State Detection using ANN</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>MediaPipe + OpenCV + Artificial Neural Network</div>",
    unsafe_allow_html=True
)

st.markdown("---")


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("👁️ Eye State Detection")

st.sidebar.success("Artificial Neural Network")

st.sidebar.info("MediaPipe Face Landmarker")

st.sidebar.info("OpenCV")

st.sidebar.markdown("---")

st.sidebar.subheader("Model")

st.sidebar.metric(
    "Algorithm",
    "ANN"
)

st.sidebar.metric(
    "Input Features",
    "256"
)

st.sidebar.metric(
    "Classes",
    "2"
)

st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "📤 Upload Image",
        "📹 Live Webcam"
    ]
)


# ==========================================================
# LOAD MEDIAPIPE
# ==========================================================

@st.cache_resource
def load_landmarker():

    return create_landmarker()


try:

    landmarker = load_landmarker()

except Exception as e:

    st.error(
        "Unable to load MediaPipe Face Landmarker."
    )

    st.code(str(e))

    st.stop()


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def process_image(image):

    """
    Complete pipeline:

    OpenCV image
        ↓
    MediaPipe
        ↓
    Eye detection
        ↓
    Eye crop
        ↓
    16 × 16 grayscale
        ↓
    256 features
    """

    features, eye_image = extract_eye_features(
        image,
        landmarker,
        eye="left"
    )

    if features is None:

        # Try right eye
        features, eye_image = extract_eye_features(
            image,
            landmarker,
            eye="right"
        )

    if features is None:

        return None, None, None

    result = predict_eye_state(
        eye_image
    )

    return result, eye_image, features


# ==========================================================
# UPLOAD IMAGE MODE
# ==========================================================

if mode == "📤 Upload Image":

    st.subheader("📤 Upload a Face Image")

    st.info(
        "Upload a clear image containing a visible face and eyes."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error(
                "Unable to read the uploaded image."
            )

            st.stop()

        # --------------------------------------------------
        # PROCESS
        # --------------------------------------------------

        result, eye_image, features = process_image(
            image
        )

        # --------------------------------------------------
        # DISPLAY ORIGINAL IMAGE
        # --------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        with col2:

            st.subheader("Prediction")

            if result is None:

                st.error(
                    "❌ No eye detected."
                )

                st.warning(
                    "Please upload a clear face image with visible eyes."
                )

            else:

                label = result["label"]

                confidence = result["confidence"]

                if label.lower() == "open":

                    st.success(
                        "👁️ EYE OPEN"
                    )

                else:

                    st.error(
                        "😴 EYE CLOSED"
                    )

                st.metric(
                    "Eye State",
                    label.upper()
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.progress(
                    float(confidence)
                )

        # --------------------------------------------------
        # EYE CROP
        # --------------------------------------------------

        if eye_image is not None:

            st.markdown("---")

            st.subheader(
                "👁️ Detected Eye"
            )

            eye_rgb = cv2.cvtColor(
                eye_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                eye_rgb,
                width=250
            )

            st.caption(
                "Eye region detected using MediaPipe"
            )

        # --------------------------------------------------
        # FEATURES
        # --------------------------------------------------

        if features is not None:

            st.markdown("---")

            st.subheader(
                "🤖 ANN Input"
            )

            st.write(
                f"Feature shape: `{features.shape}`"
            )

            st.write(
                "256 grayscale pixel features"
            )


# ==========================================================
# LIVE WEBCAM MODE
# ==========================================================

elif mode == "📹 Live Webcam":

    st.subheader(
        "📹 Live Eye State Detection"
    )

    st.info(
        "Allow camera access and keep your face clearly visible."
    )

    run_camera = st.checkbox(
        "Start Webcam"
    )

    if run_camera:

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():

            st.error(
                "❌ Could not open webcam."
            )

            st.stop()

        frame_placeholder = st.empty()

        stop_button = st.button(
            "Stop Webcam"
        )

        while camera.isOpened():

            if stop_button:
                break

            success, frame = camera.read()

            if not success:

                st.error(
                    "Unable to read webcam frame."
                )

                break

            # ------------------------------------------------
            # MIRROR IMAGE
            # ------------------------------------------------

            frame = cv2.flip(
                frame,
                1
            )

            # ------------------------------------------------
            # MEDIA PIPE + ANN
            # ------------------------------------------------

            result, eye_image, features = process_image(
                frame
            )

            # ------------------------------------------------
            # DEFAULT DISPLAY
            # ------------------------------------------------

            display_frame = frame.copy()

            if result is not None:

                label = result["label"]

                confidence = result["confidence"]

                # --------------------------------------------
                # COLOR
                # --------------------------------------------

                if label.lower() == "open":

                    color = (
                        0,
                        255,
                        0
                    )

                    status = "EYE OPEN"

                else:

                    color = (
                        0,
                        0,
                        255
                    )

                    status = "EYE CLOSED"

                # --------------------------------------------
                # TEXT
                # --------------------------------------------

                cv2.rectangle(
                    display_frame,
                    (20, 20),
                    (500, 145),
                    color,
                    3
                )

                cv2.putText(
                    display_frame,
                    f"Eye State : {status}",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Confidence : {confidence * 100:.2f}%",
                    (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    display_frame,
                    "MediaPipe + ANN",
                    (40, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            else:

                cv2.rectangle(
                    display_frame,
                    (20, 20),
                    (500, 100),
                    (0, 165, 255),
                    3
                )

                cv2.putText(
                    display_frame,
                    "No eye detected",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    "Show your face clearly",
                    (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            frame_rgb = cv2.cvtColor(
                display_frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

        camera.release()


# ==========================================================
# PROJECT OVERVIEW
# ==========================================================

st.markdown("---")

st.subheader(
    "📊 Project Overview"
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Model",
        "ANN"
    )

with c2:

    st.metric(
        "Features",
        "256"
    )

with c3:

    st.metric(
        "Computer Vision",
        "OpenCV"
    )

with c4:

    st.metric(
        "Landmark Detection",
        "MediaPipe"
    )


# ==========================================================
# ABOUT
# ==========================================================

st.markdown("---")

st.subheader(
    "ℹ️ About the Project"
)

st.write(
    """
This application detects whether an eye is **open or closed**
using a combination of **MediaPipe, OpenCV and an Artificial
Neural Network (ANN)**.

### Workflow

1. OpenCV captures the image or webcam frame.
2. MediaPipe detects facial landmarks.
3. The eye region is extracted.
4. The eye is converted to grayscale.
5. The eye is resized to 16 × 16 pixels.
6. The 256 pixel values are passed to the trained ANN.
7. The ANN predicts **Open** or **Closed**.

### Technologies

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn
- Streamlit
"""
)


# ==========================================================
# NOTE
# ==========================================================

st.markdown("---")

st.info(
    """
⚠️ **For best results**

• Use a well-lit environment.

• Keep your face visible.

• Avoid extreme side angles.

• Make sure both eyes are visible.

• For webcam detection, keep your face reasonably close
to the camera.
"""
)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    👁️ <b>Eye State Detection using MediaPipe + OpenCV + ANN</b>
    <br><br>

    Built with Python • TensorFlow • MediaPipe • OpenCV • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)