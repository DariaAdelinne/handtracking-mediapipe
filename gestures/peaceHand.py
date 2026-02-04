import cv2
from gestures.utils import load_image, overlay_image

PEACE_IMG = load_image("../assets/peace.png")


def is_peace_sign(hand_landmarks, mp_hands):
    lm = hand_landmarks.landmark

    index_up = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_up = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_down = lm[mp_hands.HandLandmark.RING_FINGER_TIP].y > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_down = lm[mp_hands.HandLandmark.PINKY_TIP].y > lm[mp_hands.HandLandmark.PINKY_PIP].y

    return index_up and middle_up and ring_down and pinky_down


class PeaceGesture:
    def __init__(self, on_frames=3, off_frames=5):
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.show = False
        self._on = 0
        self._off = 0

    def update(self, detected: bool):
        if detected:
            self._on += 1
            self._off = 0
        else:
            self._off += 1
            self._on = 0

        if not self.show and self._on >= self.on_frames:
            self.show = True
        if self.show and self._off >= self.off_frames:
            self.show = False

    def draw(self, frame):
        if self.show:
            overlay_image(frame, PEACE_IMG, 20, 20)
            cv2.putText(frame, "PEACE", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
