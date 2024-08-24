import cv2
import mediapipe as mp
import numpy as np


class HandGestureAndGenderRecognition:
    def __init__(self):
        # Load pre-trained model and configuration for gender detection
        gender_model_path = "/Users/tharun/PycharmProjects/women_safety/gender/gender_net.caffemodel"
        gender_proto_path = "/Users/tharun/PycharmProjects/women_safety/gender/gender_deploy.prototxt"
        self.gender_net = cv2.dnn.readNet(gender_model_path, gender_proto_path)

        # List of gender classes
        self.gender_list = ['Male', 'Female']

        # Initialize Mediapipe Hands and Drawing modules
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2)  # Detect up to two hands
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize face detection using OpenCV's pre-trained model (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Open the default webcam
        self.cap = cv2.VideoCapture(0)

    def calculate_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def detect_gestures(self, frame, results):
        """Detect hand gestures using Mediapipe and display the gesture type on the frame."""
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Recognize if the hand is open or closed
            wrist = hand_landmarks.landmark[0]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            # Calculate distances from wrist to fingertips
            distance_index_wrist = self.calculate_distance(index_tip, wrist)
            distance_middle_wrist = self.calculate_distance(middle_tip, wrist)
            distance_ring_wrist = self.calculate_distance(ring_tip, wrist)
            distance_pinky_wrist = self.calculate_distance(pinky_tip, wrist)

            # Determine gesture
            if (distance_index_wrist < 0.1 and
                    distance_middle_wrist < 0.1 and
                    distance_ring_wrist < 0.1 and
                    distance_pinky_wrist < 0.1):
                gesture = "Fist"
            else:
                gesture = "Open Hand"

            # Display the identified gesture on the screen
            cv2.putText(frame, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def detect_faces_and_gender(self, frame):
        """Detect faces in the frame and classify gender using the gender model."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]

            # Preprocess the face for gender classification
            blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227),
                                         mean=(78.4263377603, 87.7689143744, 114.895847746),
                                         swapRB=False, crop=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]

            # Display the gender label on the screen
            label = f"{gender}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def run(self):
        """Main method to run hand gesture and gender recognition."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe to detect hands
            results = self.hands.process(frame_rgb)

            # If hands are detected, perform gesture recognition
            if results.multi_hand_landmarks:
                self.detect_gestures(frame, results)

            # Perform face detection and gender classification
            self.detect_faces_and_gender(frame)

            # Show the frame
            cv2.imshow('Hand Gesture and Gender Recognition', frame)

            # Exit loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()


# Create an instance of the class and run the program
if __name__ == "__main__":
    recognition_system = HandGestureAndGenderRecognition()
    recognition_system.run()