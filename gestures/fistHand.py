import cv2
from gestures.utils import load_image, overlay_image

# Load the fist icon that will be displayed on screen
FIST_IMG = load_image("../assets/fist.png")


def is_fist(hand_landmarks, mp_hands):
    """
    Detects whether the current hand landmarks represent a fist gesture.
    """
    lm = hand_landmarks.landmark

    # Check if all fingers are folded (tip below the PIP joint)
    fingers_down = [
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y >
        lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,

        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y >
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,

        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y >
        lm[mp_hands.HandLandmark.RING_FINGER_PIP].y,

        lm[mp_hands.HandLandmark.PINKY_TIP].y >
        lm[mp_hands.HandLandmark.PINKY_PIP].y,
    ]

    # Check if the thumb is close to the palm (near the wrist)
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    wrist = lm[mp_hands.HandLandmark.WRIST]

    thumb_close = abs(thumb_tip.x - wrist.x) < 0.08

    return all(fingers_down) and thumb_close


class FistGesture:
    """
    Handles temporal smoothing and visualization of the fist gesture.
    """
    def __init__(self, on_frames=3, off_frames=5):
        # Number of consecutive frames required to activate/deactivate the gesture
        self.on_frames = on_frames
        self.off_frames = off_frames

        # Whether the gesture is currently shown
        self.show = False

        # Counters for gesture stability
        self._on = 0
        self._off = 0

    def update(self, detected: bool):
        """
        Updates internal state based on whether the fist was detected
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

        # Deactivate gesture after enough consecutive misses
        if self.show and self._off >= self.off_frames:
            self.show = False

    def draw(self, frame):
        """
        Draws the fist icon and label on the frame if the gesture is active.
        """
        if self.show:
            overlay_image(frame, FIST_IMG, 20, 20)
            cv2.putText(
                frame,
                "FIST",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )
