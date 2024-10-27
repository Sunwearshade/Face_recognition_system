import cv2
from recognition.detector import detect_faces
from recognition.scanner import recognize_face

def capture_and_display():
    # Inicializa la cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise Exception("No se pudo abrir la cámara")
    
    while True:
        # Captura el cuadro de la cámara
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detecta rostros en el cuadro
        face_locations = detect_faces(frame)
        
        # Recorre cada rostro detectado para verificar autorización
        for location in face_locations:
            # Verifica si el rostro está autorizado
            is_authorized = recognize_face(frame, location)
            
            # Dibuja un cuadro verde si está autorizado, rojo si no lo está
            top, right, bottom, left = location
            color = (0, 255, 0) if is_authorized else (0, 0, 255)
            label = "Autorizado" if is_authorized else "No Autorizado"
            
            # Dibuja el cuadro y etiqueta
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Muestra el cuadro en pantalla
        cv2.imshow("Face Recognition", frame)
        
        # Termina con la tecla "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
