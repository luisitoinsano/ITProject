import os
import cv2
import time
import datetime as dt
import mediapipe as mp
import numpy as np

# ---------------------------
# Configuración general
# ---------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMES_DIR = os.path.join(PROJECT_DIR, "memes")
# NUEVO: ventana
WINDOW_NAME = "MemeCam - [S] Foto  [R] Recargar memes  [Q] Salir"
# Preferencia neutral (se ajusta dinámicamente más abajo)
DEFAULT_NEUTRAL_MEME = "sigma"

# Cámara: backend, resolución y latencia
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.setUseOptimized(True)
try:
    cv2.ocl.setUseOpenCL(True)
except Exception:
    pass

# ---------------------------
# MediaPipe Holistic
# ---------------------------
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
holistic = mp.solutions.holistic.Holistic(
    model_complexity=0,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# Utilidades de memes
# ---------------------------
def scan_memes(folder):  # NUEVO: carga automática de todos los archivos de memes/
    memes = {}
    if not os.path.isdir(folder):
        print(f"[AVISO] Carpeta de memes no existe: {folder}")
        return memes
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            key = os.path.splitext(name)[0].lower()  # clave = nombre del archivo sin extensión
            path = os.path.join(folder, name)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[AVISO] No se pudo leer: {path}")
            memes[key] = img
    if memes:
        print(f"[OK] Memes cargados: {', '.join(sorted(memes.keys()))}")
    else:
        print(f"[AVISO] No se encontraron imágenes en {folder}")
    return memes

# Reemplaza load_memes/MEME_PATHS por el escaneo:
MEMES = scan_memes(MEMES_DIR)
# NUEVO: escoger neutral disponible (sigma -> sigma* -> neutral -> cualquiera)
def _pick_neutral(memes, prefer="sigma"):
    keys = list(memes.keys())
    if not keys:
        return prefer
    if prefer in memes:
        return prefer
    for k in keys:
        if k.startswith(prefer):
            return k
    if "neutral" in memes:
        return "neutral"
    return keys[0]

DEFAULT_NEUTRAL_MEME = _pick_neutral(MEMES, "sigma")
print(f"[INFO] Meme neutral seleccionado: '{DEFAULT_NEUTRAL_MEME}'")

# NUEVO: toggles de visualización
SHOW_LANDMARKS = True
SHOW_GESTURE_TEXT = True
SHOW_FACE_LANDMARKS = False  # no mostrar marcas en la cara

# NUEVO: mapeo gesto -> clave de meme (neutral usa el seleccionado)
GESTURE_TO_MEME = {
    "neutral":   DEFAULT_NEUTRAL_MEME,
    "hands_up":  "mono1",     # ejemplo: manos arriba -> mono1.jpg
    "t_pose":    "mono2",     # ejemplo: t_pose -> mono2.jpg
    "thumbs_up": "thumbs_up", # usa thumbs_up.jpg si existe
    "v_sign":    "v_sign",
    "ok_sign":   "ok_sign",
    "finger_to_mouth": "mono1",
}

def get_meme_panel(current_label, target_h, target_w=None):
    """
    Devuelve una imagen del meme redimensionada a la altura del frame de cámara.
    Si no existe el archivo, devuelve un panel placeholder con texto.
    target_w opcional: fuerza ancho si se desea un panel fijo.
    """
    meme_img = MEMES.get(current_label)
    if meme_img is None:
        # Placeholder
        w = target_w if target_w else int(target_h * 0.75)
        panel = np.zeros((target_h, w, 3), dtype=np.uint8)
        cv2.putText(panel, "Meme no disponible", (10, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return panel

    # Redimensionar manteniendo aspecto, ajustado a la altura target_h
    h, w = meme_img.shape[:2]
    scale = target_h / float(h)
    new_w = target_w if target_w else max(1, int(w * scale))
    resized = cv2.resize(meme_img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return resized

# ---------------------------
# Clasificación de gestos (reglas simples)
# ---------------------------
# Índices de dedos de MediaPipe Hands
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

def _is_extended(lms, tip_idx, pip_idx):
    # Coordenadas normalizadas; menor y => más arriba
    return lms[tip_idx].y < lms[pip_idx].y - 0.02

def _is_folded(lms, tip_idx, pip_idx):
    return lms[tip_idx].y > lms[pip_idx].y + 0.02

def _thumbs_up(hand_lms, is_left):
    # Pulgar arriba: otras falanges plegadas, pulgar apuntando hacia arriba.
    others_folded = (
        _is_folded(hand_lms, INDEX_TIP, INDEX_PIP) and
        _is_folded(hand_lms, MIDDLE_TIP, MIDDLE_PIP) and
        _is_folded(hand_lms, RING_TIP, RING_PIP) and
        _is_folded(hand_lms, PINKY_TIP, PINKY_PIP)
    )
    thumb_up = hand_lms[THUMB_TIP].y < hand_lms[THUMB_IP].y - 0.02
    return others_folded and thumb_up

def _v_sign(hand_lms):
    # Índice y medio extendidos; anular y meñique plegados
    return (
        _is_extended(hand_lms, INDEX_TIP, INDEX_PIP) and
        _is_extended(hand_lms, MIDDLE_TIP, MIDDLE_PIP) and
        _is_folded(hand_lms, RING_TIP, RING_PIP) and
        _is_folded(hand_lms, PINKY_TIP, PINKY_PIP)
    )

def _hands_up(pose_lms):
    # Ambas muñecas por encima de sus hombros
    ls = pose_lms.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    rs = pose_lms.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    lw = pose_lms.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    rw = pose_lms.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    return lw.y < ls.y - 0.03 and rw.y < rs.y - 0.03

def _t_pose(pose_lms):
    # Muñecas a la altura de hombros y separadas (brazos extendidos)
    ls = pose_lms.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    rs = pose_lms.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    lw = pose_lms.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    rw = pose_lms.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    y_ok = abs(lw.y - ls.y) < 0.06 and abs(rw.y - rs.y) < 0.06
    # Separación horizontal de muñecas vs hombros
    wrists_span = abs(rw.x - lw.x)
    shoulders_span = abs(rs.x - ls.x)
    wide = wrists_span > max(0.5, shoulders_span * 1.2)
    return y_ok and wide

def _ok_sign(hand_lms):
    """
    Gesto OK: pulgar e índice formando un círculo (sus puntas muy cercanas),
    y al menos anular y meñique plegados para evitar confundir con V-sign.
    """
    # Distancia euclidiana entre puntas de pulgar e índice (coordenadas normalizadas [0..1])
    dx = hand_lms[THUMB_TIP].x - hand_lms[INDEX_TIP].x
    dy = hand_lms[THUMB_TIP].y - hand_lms[INDEX_TIP].y
    d = (dx*dx + dy*dy) ** 0.5
    close = d < 0.06  # umbral empírico; ajústalo si hace falta
    ring_folded = _is_folded(hand_lms, RING_TIP, RING_PIP)
    pinky_folded = _is_folded(hand_lms, PINKY_TIP, PINKY_PIP)
    return close and ring_folded and pinky_folded

# NUEVO: índices boca (Face Mesh) y umbral de proximidad
MOUTH_UPPER_IDX = 13
MOUTH_LOWER_IDX = 14
MOUTH_TOUCH_THRESH = 0.055  # ajusta entre 0.045–0.07 según tu cámara

# NUEVO: utilidades para detectar dedo cerca de la boca
def _mouth_center(face_lms):
    ux, uy = face_lms.landmark[MOUTH_UPPER_IDX].x, face_lms.landmark[MOUTH_UPPER_IDX].y
    lx, ly = face_lms.landmark[MOUTH_LOWER_IDX].x, face_lms.landmark[MOUTH_LOWER_IDX].y
    return (ux + lx) * 0.5, (uy + ly) * 0.5

def _hand_tip_min_dist_to(x0, y0, hand_lms):
    # Distancia mínima entre centro de boca y puntas de índice o pulgar
    pts = [hand_lms[INDEX_TIP], hand_lms[THUMB_TIP]]
    dminsq = min((p.x - x0) ** 2 + (p.y - y0) ** 2 for p in pts)
    return dminsq ** 0.5

def _finger_to_mouth(results):
    if not results.face_landmarks:
        return False
    x0, y0 = _mouth_center(results.face_landmarks)
    # Chequear ambas manos si existen
    if results.right_hand_landmarks:
        if _hand_tip_min_dist_to(x0, y0, results.right_hand_landmarks.landmark) < MOUTH_TOUCH_THRESH:
            return True
    if results.left_hand_landmarks:
        if _hand_tip_min_dist_to(x0, y0, results.left_hand_landmarks.landmark) < MOUTH_TOUCH_THRESH:
            return True
    return False

def classify_gesture(results):
    # Prioridad: pose > dedo a la boca > manos
    if results.pose_landmarks and _hands_up(results.pose_landmarks):
        return "hands_up"
    if results.pose_landmarks and _t_pose(results.pose_landmarks):
        return "t_pose"

    # NUEVO: dedo (índice o pulgar) acercado a la boca
    if _finger_to_mouth(results):
        return "finger_to_mouth"

    left = results.left_hand_landmarks
    right = results.right_hand_landmarks

    if right and _thumbs_up(right.landmark, is_left=False):
        return "thumbs_up"
    if left and _thumbs_up(left.landmark, is_left=True):
        return "thumbs_up"

    if right and _v_sign(right.landmark):
        return "v_sign"
    if left and _v_sign(left.landmark):
        return "v_sign"

    # NUEVO: gesto OK
    if right and _ok_sign(right.landmark):
        return "ok_sign"
    if left and _ok_sign(left.landmark):
        return "ok_sign"

    return "neutral"

# ---------------------------
# Bucle principal
# ---------------------------
os.makedirs(os.path.join(PROJECT_DIR, "snapshots"), exist_ok=True)
print("Controles: [S] guardar foto  [R] recargar memes  [Q] salir")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        start_t = time.time()

        # Vista espejo para el usuario
        frame = cv2.flip(frame, 1)
        clean_bgr = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(rgb)

        # Dibujo de landmarks (condicional)
        if SHOW_LANDMARKS and SHOW_FACE_LANDMARKS and results.face_landmarks:  # CAMBIO: oculta cara por defecto
            mp_drawing.draw_landmarks(
                rgb, results.face_landmarks,
                mp.solutions.holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )
        if SHOW_LANDMARKS and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                rgb, results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )
        if SHOW_LANDMARKS and results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                rgb, results.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
        if SHOW_LANDMARKS and results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                rgb, results.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Clasificar gesto y obtener panel de meme
        gesture = classify_gesture(results)
        meme_key = GESTURE_TO_MEME.get(gesture, "neutral")
        meme_panel = get_meme_panel(meme_key, target_h=vis_bgr.shape[0])

        # Componer lado a lado
        combined = cv2.hconcat([vis_bgr, meme_panel])
        combined_clean = cv2.hconcat([clean_bgr, meme_panel])

        # Overlay condicional: nombre del gesto y FPS
        if SHOW_GESTURE_TEXT:
            fps = 1.0 / max(1e-6, time.time() - start_t)
            cv2.putText(combined, f"Gesto: {gesture}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)
            cv2.putText(combined, f"FPS: {fps:.1f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Guardar foto limpia con nombres claros
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(PROJECT_DIR, "snapshots", f"snapshot_{gesture}_{meme_key}_{ts}.png")
            cv2.imwrite(out_path, combined_clean)
            print(f"[OK] Foto guardada: {out_path}")
        elif key == ord('r'):
            # Recargar imágenes de memes (escaneo automático de la carpeta)
            MEMES = scan_memes(MEMES_DIR)  # CAMBIO
            print("[OK] Memes recargados.")
        elif key == ord('l'):  # NUEVO: alternar landmarks
            SHOW_LANDMARKS = not SHOW_LANDMARKS
        elif key == ord('o'):  # NUEVO: alternar textos Gesto/FPS
            SHOW_GESTURE_TEXT = not SHOW_GESTURE_TEXT

finally:
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
