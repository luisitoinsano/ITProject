import cv2
from fer.fer import FER
import mediapipe as mp

# Paleta de colores (BGR) y etiquetas en español por emoción
EMOTION_COLORS = {
    "angry": (0, 0, 255),       # rojo
    "disgust": (0, 128, 0),     # verde oscuro
    "fear": (128, 0, 128),      # morado
    "happy": (0, 255, 255),     # amarillo
    "sad": (255, 0, 0),         # azul
    "surprise": (255, 255, 0),  # cian
    "neutral": (200, 200, 200)  # gris
}
EMOTION_LABELS = {
    "angry": "Enojo",
    "disgust": "Asco",
    "fear": "Miedo",
    "happy": "Alegría",
    "sad": "Tristeza",
    "surprise": "Sorpresa",
    "neutral": "Neutro"
}

# Inicializa la cámara
cap = cv2.VideoCapture(0)

# Crea un detector de emociones
detector = FER(mtcnn=True)

# MediaPipe: dibujo y modelo Holistic (cara, manos y pose)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB antes de detectar (mejora la precisión del modelo)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe Holistic: procesar y dibujar landmarks (cara, manos, pose) en RGB
    results = holistic.process(rgb_frame)

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            rgb_frame,
            results.face_landmarks,
            mp.solutions.holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
            rgb_frame,
            results.face_landmarks,
            mp.solutions.holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            rgb_frame,
            results.pose_landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_l