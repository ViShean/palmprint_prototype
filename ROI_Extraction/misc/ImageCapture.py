import cv2
import time
import os
import numpy as np
from datetime import datetime

# Create folder: Images/YYYY-MM-DD_HH-MM-SS
current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
image_folder = os.path.join("Images", current_timestamp)
os.makedirs(image_folder, exist_ok=True)

# Open the default camera
cap = cv2.VideoCapture(0)

# Set video properties (Optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4608)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2592)

start_time = time.time()
record_duration = 5  # Seconds
frame_interval = 0.2  # 5 FPS

last_second = int(start_time)
frame_counter = 1

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
        break

    # Check for black frames (less sensitive)
    if np.mean(frame) < 10: #changed from np.all(frame == 0)
        print(f"Skipping dark frame at second {current_second}, frame {frame_counter}")
        continue

    if frame_counter <= 5:
        image_filename = os.path.join(image_folder, f"second_{current_second}_{frame_counter}.jpg")
        cv2.imwrite(image_filename, frame)
        print(f"Image saved: {image_filename}")
        frame_counter += 1

    # Calculate the time to wait until the next frame
    next_frame_time = start_time + (current_second - 1 + (frame_counter - 1) * frame_interval)
    wait_time = max(0, next_frame_time - time.time())
    time.sleep(wait_time)

cap.release()
cv2.destroyAllWindows()

print(f"All images saved in: {image_folder}")