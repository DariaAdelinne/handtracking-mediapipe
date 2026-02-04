import cv2
import mediapipe as mp

from gestures.stopHand import StopGesture, is_open_palm
from gestures.peaceHand import PeaceGesture, is_peace_sign
from gestures.fistHand import FistGesture, is_fist
from gestures.oneFingerHand import OneFingerGesture, is_one_finger
from gestures.thumbsUpHand import ThumbsUpGesture, is_thumbs_up
from gestures.thumbsDownHand import ThumbsDownGesture, is_thumbs_down


# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def main():
    # Open default webcam
    cap = cv2.VideoCapture(0)

    # Initialize gesture handlers
    stop_gesture = StopGesture(on_frames=3, off_frames=5)
    peace_gesture = PeaceGesture(on_frames=3, off_frames=5)
    fist_gesture = FistGesture(on_frames=3, off_frames=5)
    one_gesture = OneFingerGesture(on_frames=3, off_frames=5)
    thumbs_up_gesture = ThumbsUpGesture(on_frames=3, off_frames=5)
    thumbs_down_gesture = ThumbsDownGesture(on_frames=3, off_frames=5)

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror image for natural interaction
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Detection flags for each gesture
            detected_stop = False
            detected_peace = False
            detected_fist = False
            detected_one = False
            detected_thumbs_up = False
            detected_thumbs_down = False

            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]

                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                # Run gesture detection functions
                detected_stop = is_open_palm(hand_lm, mp_hands)
                detected_peace = is_peace_sign(hand_lm, mp_hands)
                detected_fist = is_fist(hand_lm, mp_hands)
                detected_one = is_one_finger(hand_lm, mp_hands)
                detected_thumbs_up = is_thumbs_up(hand_lm, mp_hands)
                detected_thumbs_down = is_thumbs_down(hand_lm, mp_hands)

            # Update gesture state machines
            stop_gesture.update(detected_stop)
            peace_gesture.update(detected_peace)
            fist_gesture.update(detected_fist)
            one_gesture.update(detected_one)
            thumbs_up_gesture.update(detected_thumbs_up)
            thumbs_down_gesture.update(detected_thumbs_down)

            # Gesture rendering priority (highest first)
            if thumbs_up_gesture.show:
                thumbs_up_gesture.draw(frame)
            elif thumbs_down_gesture.show:
                thumbs_down_gesture.draw(frame)
            elif one_gesture.show:
                one_gesture.draw(frame)
            elif fist_gesture.show:
                fist_gesture.draw(frame)
            elif peace_gesture.show:
                peace_gesture.draw(frame)
            elif stop_gesture.show:
                stop_gesture.draw(frame)

            # Display the result
            cv2.imshow("Hand Tracking", frame)

            # Exit on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
