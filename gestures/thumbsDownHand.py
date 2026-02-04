import cv2
from .utils import load_image, overlay_image

# Load the icon that represents the thumbs-down gesture
THUMBS_DOWN_IMG = load_image("thumbs_down.png")


def is_thumbs_down(hand_landmarks, mp_hands):
    """
    Detects whether the current hand landmarks represent
    a thumbs-down gesture.
    """
    lm = hand_landmarks.landmark

    # Thumb is pointing downward (tip below the IP joint)
    thumb_down = (
        lm[mp_hands.HandLandmark.THUMB_TIP].y >
        lm[mp_hands.HandLandmark.THUMB_IP].y
    )

    # Other fingers are folded
    index_down = (
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y >
        lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    )
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

    # Extra stabilization: thumb tip is below the base of the index finger
    thumb_below_index_base = (
        lm[mp_hands.HandLandmark.THUMB_TIP].y >
        lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    )

    return (
        thumb_down and
        thumb_below_index_base and
        index_down and
        middle_down and
        ring_down and
        pinky_down
    )


class ThumbsDownGesture:
    """
    Handles temporal smoothing and visualization
    of the thumbs-down gesture.
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
            overlay_image(frame, THUMBS_DOWN_IMG, 20, 20)
            cv2.putText(
                frame,
                "THUMBS DOWN",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )
