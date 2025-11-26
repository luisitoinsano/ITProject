import os
import cv2
from django.http import StreamingHttpResponse, HttpResponse, FileResponse
from django.shortcuts import render
from django.conf import settings

try:
    import mediapipe as mp
except Exception:
    mp = None

BASE_DIR = settings.BASE_DIR if hasattr(settings, 'BASE_DIR') else os.getcwd()

def index(request):
    return render(request, 'viewer/index.html')


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
            if holistic:
                results = holistic.process(rgb)
                annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks and drawing_utils:
                    drawing_utils.draw_landmarks(annotated, results.pose_landmarks, pose_connections)
            else:
                annotated = frame

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
