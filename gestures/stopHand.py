import cv2
from gestures.utils import load_image, overlay_image

# Load the icon that represents the open palm / stop gesture
STOP_IMG = load_image("../assets/stop.png")


def is_open_palm(hand_landmarks, mp_hands):
    """
    Detects whether the current hand landmarks represent
    an open palm (stop) gesture.
    """
    lm = hand_landmarks.landmark

    # All four fingers are extended
    fingers_up = [
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
        lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,

        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y <
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,

        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y <
        lm[mp_hands.HandLandmark.RING_FINGER_PIP].y,

        lm[mp_hands.HandLandmark.PINKY_TIP].y <
        lm[mp_hands.HandLandmark.PINKY_PIP].y,
    ]

    # Thumb is spread away from the palm (used to distinguish from a fist)
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    wrist = lm[mp_hands.HandLandmark.WRIST]
    thumb_distance = abs(thumb_tip.x - wrist.x)

    return all(fingers_up) and thumb_distance > 0.10


class StopGesture:
    """
    Handles temporal smoothing and visualization
    of the stop (open palm) gesture.
    """
    def __init__(self, on_frames=3, off_frames=5):
        # Number of consecutive frames required to activate/deactivate the gesture
        self.on_frames = on_frames
        self.off_frames = off_frames

        # Whether the gesture is currently active
        self.show = False

        # Frame counters for gesture stability
        self._on = 0
        self._off = 0

    def update(self, detected: bool):
        """
        Updates gesture state based on detection
        in the current frame.
        """
        if detected:
            self._on += 1
            self._off = 0
        else:
            self._off += 1
            self._on = 0

        # Activate gesture after enough consecutive detections
        if not self.show and self._on >= self.on_frames:
            self.show = True

        # Deactivate gesture after enough consecutive missed detections
        if self.show and self._off >= self.off_frames:
            self.show = False

    def draw(self, frame):
        """
        Draws the gesture icon and label if the gesture is active.
        """
        if self.show:
            overlay_image(frame, STOP_IMG, 20, 20)
            cv2.putText(
                frame,
                "STOP",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )
