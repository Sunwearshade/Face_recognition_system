import face_recognition
import cv2
import pickle
import os
import numpy as np

AUTHORIZED_FACES_PATH = "data/authorized_faces.pkl"
TOLERANCE = 0.5  # Ajusta la tolerancia para mayor precisión
CONFIDENCE_THRESHOLD = 0.6  # Umbral de confianza para clasificar como autorizado

def register_authorized_face():
    cap = cv2.VideoCapture(0)
    face_encodings = []
    print("Mira a la cámara. Se capturarán varias muestras de tu rostro para aumentar la precisión.")
    
    while len(face_encodings) < 5:  # Capturamos 5 muestras
        ret, frame = cap.read()
        
        if not ret:
            break
        
        face_locations = face_recognition.face_locations(frame)
        
        if face_locations:
            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
            face_encodings.append(face_encoding)
            print(f"Muestra {len(face_encodings)} capturada.")
    
    # Guarda las facciones del rostro autorizado
    with open(AUTHORIZED_FACES_PATH, "wb") as f:
        pickle.dump(face_encodings, f)
    
    print("Registro completado. Solo este rostro tendrá acceso autorizado.")
    cap.release()
    cv2.destroyAllWindows()

def recognize_face(frame, face_location):
    if not os.path.exists(AUTHORIZED_FACES_PATH):
        print("No se encontraron rostros autorizados. Regístralos primero.")
        return False
    
    with open(AUTHORIZED_FACES_PATH, "rb") as f:
        authorized_faces = pickle.load(f)
    
    # Extrae las facciones del rostro detectado
    face_encoding = face_recognition.face_encodings(frame, [face_location])[0]
    
    # Calcula la distancia entre el rostro detectado y cada rostro autorizado
    distances = face_recognition.face_distance(authorized_faces, face_encoding)
    
    # Obtiene la coincidencia más cercana
    min_distance = np.min(distances)
    
    # Verifica s la distancia mínima está dentro del umbral
    if min_distance < TOLERANCE:
        # Verifica si la confianza está por encima del umbral
        if (1 - min_distance) > CONFIDENCE_THRESHOLD:
            return True  # Rostro autorizado
        else:
            return False  # No autorizado
    return False  # No autorizado
