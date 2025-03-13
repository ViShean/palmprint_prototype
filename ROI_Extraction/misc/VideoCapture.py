import cv2
import time
import os
from datetime import datetime

# Create 'video' folder if it doesn't exist
output_folder = "video"
os.makedirs(output_folder, exist_ok=True)

# Generate a filename with date and time
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_filename = os.path.join(output_folder, f"video_{timestamp}.mp4")

# Open the default camera (0 for built-in or USB webcam)
cap = cv2.VideoCapture(0)

# Set video properties (Optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Define the codec and VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is for MP4 format
out = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))

start_time = time.time()
record_duration = 5  # Seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the recording frame (Optional)
    cv2.imshow('Recording', frame)
    
    # Write frame to video file
    out.write(frame)

    # Break the loop after 5 seconds
    if time.time() - start_time > record_duration:
        break

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved to: {video_filename}")