import cv2
import mediapipe as mp
import numpy as np
import math

# Setup Mediapipe instance
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

# Push-up counter variables
counter = 0
stage = None

# Define angle calculation function
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (elbow)
    c = np.array(c)  # Last point

    # Compute vectors
    ab = a - b
    bc = c - b

    # Calculate angle using the dot product formula
    dot = np.dot(ab, bc)
    cross = np.cross(ab, bc)
    angle = math.atan2(np.linalg.norm(cross), dot)

    # Convert angle to degrees
    angle = math.degrees(angle)

    return angle

# Start pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for key landmarks: shoulder, elbow, wrist
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle of elbow
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize the angle on the frame
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Push-up counter logic
            if angle > 170:  # When the arms are fully extended (push-up "up" phase)
                stage = "up"
            if angle < 90 and stage == 'up':  # When the elbow angle is small (push-up "down" phase)
                stage = "down"
                counter += 1  # Increment rep counter
                print(f"Push-ups: {counter}")

        except:
            pass

        # Create a smooth, rounded background for the counter UI elements
        cv2.rectangle(image, (0, 0), (250, 100), (35, 40, 45), -1)  # Dark background for the counter area
        cv2.rectangle(image, (0, 100), (250, 200), (35, 40, 45), -1)  # Dark background for the stage area

        # Add some smooth rounded corners to the boxes
        rounded_box = np.zeros_like(image)
        cv2.ellipse(rounded_box, (20, 20), (20, 20), 0, 0, 180, (0, 0, 0), -1)
        cv2.ellipse(rounded_box, (230, 20), (20, 20), 0, 0, 180, (0, 0, 0), -1)
        cv2.ellipse(rounded_box, (20, 180), (20, 20), 0, 180, 360, (0, 0, 0), -1)
        cv2.ellipse(rounded_box, (230, 180), (20, 20), 0, 180, 360, (0, 0, 0), -1)

        image = cv2.addWeighted(image, 1, rounded_box, 0.7, 0)

        # Rep data with updated styling
        cv2.putText(image, 'REPS', (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        # Stage data with updated styling
        cv2.putText(image, 'STAGE', (65, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, stage if stage else "N/A",
                    (60, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        # Render detections (pose landmarks)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Show the image with push-up counter and landmarks
        cv2.imshow('Push-up Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
