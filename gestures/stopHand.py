import cv2
from gestures.utils import load_image, overlay_image

STOP_IMG = load_image("../assets/stop.png")


def is_open_palm(hand_landmarks, mp_hands):
    lm = hand_landmarks.landmark

    fingers_up = [
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y,
    ]

    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    wrist = lm[mp_hands.HandLandmark.WRIST]
    thumb_distance = abs(thumb_tip.x - wrist.x)

    return all(fingers_up) and thumb_distance > 0.10


class StopGesture:
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
            overlay_image(frame, STOP_IMG, 20, 20)
            cv2.putText(frame, "STOP", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
