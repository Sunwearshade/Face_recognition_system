from camera.camera import capture_and_display
from recognition.scanner import register_authorized_face, recognize_face

def main():
    register_authorized_face()
    
    capture_and_display()

if __name__ == "__main__":
    main()