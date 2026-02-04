import cv2
from .utils import load_image, overlay_image

THUMBS_DOWN_IMG = load_image("thumbs_down.png")


def is_thumbs_down(hand_landmarks, mp_hands):
    lm = hand_landmarks.landmark

    # Thumb "down": tip mai jos decât IP
    thumb_down = lm[mp_hands.HandLandmark.THUMB_TIP].y > lm[mp_hands.HandLandmark.THUMB_IP].y

    # Celelalte degete îndoite
    index_down = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_down = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_down = lm[mp_hands.HandLandmark.RING_FINGER_TIP].y > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_down = lm[mp_hands.HandLandmark.PINKY_TIP].y > lm[mp_hands.HandLandmark.PINKY_PIP].y

    # Extra stabilizare: thumb tip sub index MCP
    thumb_below_index_base = lm[mp_hands.HandLandmark.THUMB_TIP].y > lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

    return thumb_down and thumb_below_index_base and index_down and middle_down and ring_down and pinky_down


class ThumbsDownGesture:
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
            overlay_image(frame, THUMBS_DOWN_IMG, 20, 20)
            cv2.putText(frame, "THUMBS DOWN", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
