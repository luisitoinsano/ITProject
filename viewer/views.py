import os
import cv2
import json
from django.http import StreamingHttpResponse, HttpResponse, FileResponse, JsonResponse
from django.shortcuts import render
from django.conf import settings

try:
    import mediapipe as mp
except Exception:
    mp = None

BASE_DIR = settings.BASE_DIR if hasattr(settings, 'BASE_DIR') else os.getcwd()

# Shared state: current meme key (updated by the streaming generator)
current_meme = {'key': 'neutral'}


def index(request):
    return render(request, 'viewer/index.html')


# ---------------------------
# Simple gesture classifier copied from realtime_fer.py (small subset)
# ---------------------------
# Fingers indices
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

def _is_extended(lms, tip_idx, pip_idx):
    return lms[tip_idx].y < lms[pip_idx].y - 0.02

def _is_folded(lms, tip_idx, pip_idx):
    return lms[tip_idx].y > lms[pip_idx].y + 0.02

def _thumbs_up(hand_lms, is_left):
    others_folded = (
        _is_folded(hand_lms, INDEX_TIP, INDEX_PIP) and
        _is_folded(hand_lms, MIDDLE_TIP, MIDDLE_PIP) and
        _is_folded(hand_lms, RING_TIP, RING_PIP) and
        _is_folded(hand_lms, PINKY_TIP, PINKY_PIP)
    )
    thumb_up = hand_lms[THUMB_TIP].y < hand_lms[THUMB_IP].y - 0.02
    return others_folded and thumb_up

def _v_sign(hand_lms):
    return (
        _is_extended(hand_lms, INDEX_TIP, INDEX_PIP) and
        _is_extended(hand_lms, MIDDLE_TIP, MIDDLE_PIP) and
        _is_folded(hand_lms, RING_TIP, RING_PIP) and
        _is_folded(hand_lms, PINKY_TIP, PINKY_PIP)
    )

def _hands_up(pose_lms):
    ls = pose_lms.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    rs = pose_lms.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    lw = pose_lms.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    rw = pose_lms.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    return lw.y < ls.y - 0.03 and rw.y < rs.y - 0.03

def _t_pose(pose_lms):
    ls = pose_lms.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    rs = pose_lms.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    lw = pose_lms.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    rw = pose_lms.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    y_ok = abs(lw.y - ls.y) < 0.06 and abs(rw.y - rs.y) < 0.06
    wrists_span = abs(rw.x - lw.x)
    shoulders_span = abs(rs.x - ls.x)
    wide = wrists_span > max(0.5, shoulders_span * 1.2)
    return y_ok and wide

def _ok_sign(hand_lms):
    dx = hand_lms[THUMB_TIP].x - hand_lms[INDEX_TIP].x
    dy = hand_lms[THUMB_TIP].y - hand_lms[INDEX_TIP].y
    d = (dx*dx + dy*dy) ** 0.5
    close = d < 0.06
    ring_folded = _is_folded(hand_lms, RING_TIP, RING_PIP)
    pinky_folded = _is_folded(hand_lms, PINKY_TIP, PINKY_PIP)
    return close and ring_folded and pinky_folded

MOUTH_UPPER_IDX = 13
MOUTH_LOWER_IDX = 14
MOUTH_TOUCH_THRESH = 0.055

def _mouth_center(face_lms):
    ux, uy = face_lms.landmark[MOUTH_UPPER_IDX].x, face_lms.landmark[MOUTH_UPPER_IDX].y
    lx, ly = face_lms.landmark[MOUTH_LOWER_IDX].x, face_lms.landmark[MOUTH_LOWER_IDX].y
    return (ux + lx) * 0.5, (uy + ly) * 0.5

def _hand_tip_min_dist_to(x0, y0, hand_lms):
    pts = [hand_lms[INDEX_TIP], hand_lms[THUMB_TIP]]
    dminsq = min((p.x - x0) ** 2 + (p.y - y0) ** 2 for p in pts)
    return dminsq ** 0.5

def _finger_to_mouth(results):
    if not results.face_landmarks:
        return False
    x0, y0 = _mouth_center(results.face_landmarks)
    if results.right_hand_landmarks:
        if _hand_tip_min_dist_to(x0, y0, results.right_hand_landmarks.landmark) < MOUTH_TOUCH_THRESH:
            return True
    if results.left_hand_landmarks:
        if _hand_tip_min_dist_to(x0, y0, results.left_hand_landmarks.landmark) < MOUTH_TOUCH_THRESH:
            return True
    return False

GESTURE_TO_MEME = {
    'neutral': 'sigma1',
    'hands_up': 'mono1',
    't_pose': 'mono2',
    'thumbs_up': 'thumbs_up',
    'v_sign': 'v_sign',
    'ok_sign': 'ok_sign',
    'finger_to_mouth': 'mono1',
}

def classify_gesture(results):
    if results.pose_landmarks and _hands_up(results.pose_landmarks):
        return 'hands_up'
    if results.pose_landmarks and _t_pose(results.pose_landmarks):
        return 't_pose'
    if _finger_to_mouth(results):
        return 'finger_to_mouth'
    left = results.left_hand_landmarks
    right = results.right_hand_landmarks
    if right and _thumbs_up(right.landmark, is_left=False):
        return 'thumbs_up'
    if left and _thumbs_up(left.landmark, is_left=True):
        return 'thumbs_up'
    if right and _v_sign(right.landmark):
        return 'v_sign'
    if left and _v_sign(left.landmark):
        return 'v_sign'
    if right and _ok_sign(right.landmark):
        return 'ok_sign'
    if left and _ok_sign(left.landmark):
        return 'ok_sign'
    return 'neutral'


def gen_frames():
    # Simple MJPEG generator using MediaPipe holistics if available
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if mp:
        holistic = mp.solutions.holistic.Holistic(
            model_complexity=0, refine_face_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        drawing_utils = mp.solutions.drawing_utils
        pose_connections = mp.solutions.holistic.POSE_CONNECTIONS
    else:
        holistic = None
        drawing_utils = None
        pose_connections = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated = frame
            results = None
            if holistic:
                results = holistic.process(rgb)
                annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks and drawing_utils:
                    drawing_utils.draw_landmarks(annotated, results.pose_landmarks, pose_connections)

            # Update shared meme state using a lightweight classifier if mediapipe produced results
            try:
                if results is not None:
                    gesture = classify_gesture(results)
                    meme_key = GESTURE_TO_MEME.get(gesture, 'neutral')
                    current_meme['key'] = meme_key
            except Exception:
                # keep previous meme on errors
                pass

            ret, buffer = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        if holistic:
            holistic.close()


def stream_view(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def stream_page(request):
    # render page that shows the MJPEG stream and the current meme side-by-side
    meme = current_meme.get('key', 'neutral')
    return render(request, 'viewer/stream_page.html', {'meme': meme})


def current_meme_api(request):
    # Return JSON with the current meme key; front-end will build the URL to /media/memes/<key>.(ext) by trying common extensions
    return JsonResponse({'meme': current_meme.get('key', 'neutral')})



def code_view(request):
    path = os.path.join(BASE_DIR, 'realtime_fer.py')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            code = f.read()
        return render(request, 'viewer/code.html', {'code': code})
    return HttpResponse('realtime_fer.py not found', status=404)


def _list_files(subdir):
    folder = os.path.join(BASE_DIR, subdir)
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return files


def memes_view(request):
    imgs = _list_files('memes')
    return render(request, 'viewer/gallery.html', {'title': 'Memes', 'files': imgs, 'prefix': '/media/memes/'})


def snapshots_view(request):
    imgs = _list_files('snapshots')
    return render(request, 'viewer/gallery.html', {'title': 'Snapshots', 'files': imgs, 'prefix': '/media/snapshots/'})


def download_requirements(request):
    path = os.path.join(BASE_DIR, 'requirements.txt')
    if os.path.exists(path):
        return FileResponse(open(path, 'rb'), as_attachment=True, filename='requirements.txt')
    return HttpResponse('requirements.txt not found', status=404)
