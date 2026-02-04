import cv2
from gestures.utils import load_image, overlay_image

# Load the icon that represents the peace (V) gesture
PEACE_IMG = load_image("../assets/peace.png")


def is_peace_sign(hand_landmarks, mp_hands):
    """
    Detects whether the current hand landmarks represent
    a peace (V) sign gesture.
    """
    lm = hand_landmarks.landmark

    # Index and middle fingers are extended
    index_up = (
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
        lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    )
    middle_up = (
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y <
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    )

    # Ring and pinky fingers are folded
    ring_down = (
        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y >
        lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    )
    pinky_down = (
        lm[mp_hands.HandLandmark.PINKY_TIP].y >
        lm[mp_hands.HandLandmark.PINKY_PIP].y
    )

    return index_up and middle_up and ring_down and pinky_down


class PeaceGesture:
    """
    Handles temporal smoothing and visualization
    of the peace gesture.
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
            overlay_image(frame, PEACE_IMG, 20, 20)
            cv2.putText(
                frame,
                "PEACE",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )
