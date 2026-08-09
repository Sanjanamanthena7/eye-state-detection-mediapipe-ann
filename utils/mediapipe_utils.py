from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ==========================================================
# PATH
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TASK_FILE = BASE_DIR / "models" / "face_landmarker.task"


# ==========================================================
# MEDIAPIPE
# ==========================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


# ==========================================================
# CREATE LANDMARKER
# ==========================================================

def create_landmarker():

    if not TASK_FILE.exists():
        raise FileNotFoundError(
            f"MediaPipe task file not found:\n{TASK_FILE}"
        )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(TASK_FILE)
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    return FaceLandmarker.create_from_options(options)


# ==========================================================
# EYE LANDMARK INDEXES
# ==========================================================

# MediaPipe Face Mesh / Face Landmarker eye regions
LEFT_EYE = [
    362, 385, 387, 263,
    373, 380
]

RIGHT_EYE = [
    33, 160, 158, 133,
    153, 144
]


# ==========================================================
# GET PIXEL COORDINATES
# ==========================================================

def get_eye_points(
    face_landmarks,
    image_width,
    image_height,
    indexes
):

    points = []

    for index in indexes:

        landmark = face_landmarks[index]

        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)

        points.append((x, y))

    return points


# ==========================================================
# CROP EYE
# ==========================================================

def crop_eye(
    image,
    points,
    padding=15
):

    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = max(min(xs) - padding, 0)
    y1 = max(min(ys) - padding, 0)

    x2 = min(max(xs) + padding, image.shape[1])
    y2 = min(max(ys) + padding, image.shape[0])

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


# ==========================================================
# EXTRACT EYE FROM IMAGE
# ==========================================================

def extract_eye(
    image,
    landmarker,
    eye="left"
):

    if image is None:
        return None

    height, width = image.shape[:2]

    # OpenCV BGR → RGB
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect face landmarks
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    face_landmarks = result.face_landmarks[0]

    if eye == "left":
        indexes = LEFT_EYE
    else:
        indexes = RIGHT_EYE

    points = get_eye_points(
        face_landmarks,
        width,
        height,
        indexes
    )

    eye_image = crop_eye(
        image,
        points
    )

    return eye_image


# ==========================================================
# PREPARE EYE FOR ANN
# ==========================================================

def prepare_eye_for_ann(eye_image):

    if eye_image is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(
        eye_image,
        cv2.COLOR_BGR2GRAY
    )

    # Same dimensions used by ANN
    gray = cv2.resize(
        gray,
        (16, 16),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    gray = gray.astype(
        np.float32
    ) / 255.0

    # Flatten
    features = gray.reshape(
        1,
        256
    )

    return features


# ==========================================================
# COMPLETE MEDIA PIPELINE
# ==========================================================

def extract_eye_features(
    image,
    landmarker,
    eye="left"
):

    eye_image = extract_eye(
        image,
        landmarker,
        eye=eye
    )

    if eye_image is None:
        return None, None

    features = prepare_eye_for_ann(
        eye_image
    )

    return features, eye_image