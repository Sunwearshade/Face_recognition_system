import cv2

def resize_frame(frame, width=500):
    ratio = width / frame.shape[1]
    dim = (width, int(frame.shape[0] * ratio))
    return cv2.resize(frame, dim)
