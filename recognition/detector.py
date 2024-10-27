# recognition/detector.py

import face_recognition

def detect_faces(frame):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    return face_locations
