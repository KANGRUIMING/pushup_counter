# Push-Up Counter with OpenCV and Mediapipe

This project uses OpenCV and Mediapipe to count push-ups in real-time using computer vision. The program detects the user's body posture and tracks their arm movements to determine the number of push-ups performed.

## Features

- **Real-Time Push-Up Detection**: Counts push-ups based on elbow angle changes using the Mediapipe pose estimation model.
- **Visual Feedback**: Displays real-time feedback with the number of push-ups completed and the current stage ("up" or "down").
- **Pose Landmarks**: Visualizes key body landmarks like the shoulder, elbow, and wrist for better feedback.
- **Simple Interface**: Use the webcam to track movements and display the results.

## Requirements

- Python 3.7+
- OpenCV
- Mediapipe
- NumPy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/push-up-counter.git
   cd push-up-counter
bash```

2. Install the required dependencies:

   ```bash
   pip install opencv-python mediapipe numpy
  bash```

3. Ensure that you have a webcam connected to your system.
4. Run python app.py
5. Code Explanation
Key Components:
Pose Detection:
The script uses the Mediapipe library to detect and track the user's body landmarks (e.g., shoulder, elbow, wrist).

Angle Calculation:
The angle between the shoulder, elbow, and wrist is calculated using the dot product and cross product formula to determine the arm position.

Push-Up Counter:
The counter increments when the angle between the shoulder, elbow, and wrist goes from "up" (full extension) to "down" (elbow bent).

User Interface:
The current count of push-ups and the push-up stage ("up" or "down") is displayed in the webcam feed.
