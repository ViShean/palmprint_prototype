import cv2
import time
import os
import numpy as np
import subprocess
from datetime import datetime

# Function to calculate clarity using Laplacian variance
def calculate_clarity(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

# Create folder: Images/YYYY-MM-DD_HH-MM-SS
current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
image_folder = os.path.join("Images", current_timestamp)
os.makedirs(image_folder, exist_ok=True)

# Open the default camera, change according to the camera index
cap = cv2.VideoCapture(0)

# Set video properties (Optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Reduced resolution for stability
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get actual resolution and FPS
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")

start_time = time.time()
record_duration = 6  # Seconds
frame_interval = 0.2  # 5 FPS

last_second = int(start_time)
frame_counter = 1

# Open the text file to save clarity scores
clarity_file = os.path.join(image_folder, "clarity_scores.txt")
with open(clarity_file, "w") as f:
    f.write("Frame\tClarity Score\n")  # Write header for the clarity scores

while time.time() - start_time < record_duration:
    elapsed_time = time.time() - start_time
    current_second = int(elapsed_time)
    
    if current_second == 0:
        continue

    if current_second != last_second:
        frame_counter = 1
        last_second = current_second

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display preview with clarity score
    clarity_score = calculate_clarity(frame)
    cv2.putText(frame, f"Clarity: {clarity_score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Time: {current_second}s", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Camera Preview', frame)

    if frame_counter <= 5:
        image_filename = os.path.join(image_folder, f"second_{current_second}_{frame_counter}.jpg")
        cv2.imwrite(image_filename, frame)
        
        # Print clarity score when the image is saved
        print(f"Image saved: {image_filename}, Clarity Score: {clarity_score:.2f}")
        
        # Save the clarity score to the text file
        with open(clarity_file, "a") as f:
            f.write(f"second_{current_second}_{frame_counter}.jpg\t{clarity_score:.2f}\n")
        
        frame_counter += 1

    # Calculate the time to wait until the next frame
    next_frame_time = start_time + (current_second - 1 + (frame_counter - 1) * frame_interval)
    wait_time = max(0, next_frame_time - time.time())
    time.sleep(wait_time)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"All images saved in: {image_folder}")
print(f"Clarity scores saved in: {clarity_file}")

subprocess.run(["python", "ROIExtractionScript.py", image_folder])