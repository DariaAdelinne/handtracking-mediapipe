import cv2
from gestures.utils import load_image, overlay_image

# Load the icon that represents the "one finger" gesture
ONE_IMG = load_image("../assets/one.png")


def is_one_finger(hand_landmarks, mp_hands):
    """
    Detects whether the current hand landmarks represent
    a single raised index finger.
    """
    lm = hand_landmarks.landmark

    # Index finger is extended (tip above the PIP joint)
    index_up = (
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
        lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    )

    # Other fingers are folded
    middle_down = (
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y >
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    )
    ring_down = (
        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y >
        lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    )
    pinky_down = (
        lm[mp_hands.HandLandmark.PINKY_TIP].y >
        lm[mp_hands.HandLandmark.PINKY_PIP].y
    )

    # Thumb is ignored and does not affect detection
    return index_up and middle_down and ring_down and pinky_down


class OneFingerGesture:
    """
    Handles temporal smoothing and visualization
    of the one-finger gesture.
    """
    def __init__(self, on_frames=3, off_frames=5):
        # Number of consecutive frames required to turn the gesture on/off
        self.on_frames = on_frames
        self.off_frames = off_frames

        # Whether the gesture is currently active
        self.show = False

        # Frame counters for stability
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
            overlay_image(frame, ONE_IMG, 20, 20)
            cv2.putText(
                frame,
                "ONE",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )
