import cv2
import time
import os
from datetime import datetime
import numpy as np
import subprocess

def calculate_sharpness(image):
    """Calculate sharpness using Laplacian variance."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

# --- Video Recording Section ---
output_folder = "video"
os.makedirs(output_folder, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_filename = os.path.join(output_folder, f"video_{timestamp}.mp4")

# Initialize camera with lower resolution for better FPS
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Reduced from 1920
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Reduced from 1080

# Get ACTUAL camera FPS
requested_fps = 30
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera's actual FPS: {actual_fps}")

# Use frame-count based recording instead of time-based
record_duration = 5  # Seconds
target_frames = int(record_duration * actual_fps)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, actual_fps, 
                     (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
start_time = time.time()

while frame_count < target_frames:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow('Video Preview', frame)
    frame_count += 1

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate actual performance
real_duration = time.time() - start_time
print(f"Requested: {record_duration}s, Actual: {real_duration:.2f}s")
print(f"Target frames: {target_frames}, Captured: {frame_count}")

cap.release()
out.release()
cv2.destroyAllWindows()

# --- Frame Extraction Section ---
video_name = os.path.splitext(os.path.basename(video_filename))[0]
frames_folder = os.path.join("frames", video_name)
os.makedirs(frames_folder, exist_ok=True)

cap = cv2.VideoCapture(video_filename)

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps  # Duration in seconds, can be fractional

print(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration} sec")

# Open a text file to save clarity scores
clarity_file = os.path.join(frames_folder, "clarity_scores.txt")
with open(clarity_file, "w") as f:
    f.write("Second\tFrame\tSharpness Score\n")  # Write header for the clarity scores

# Loop through each second of video
for sec in range(int(duration)):  # Loop through each second
    start_frame = sec * fps
    frame_scores = []  # List to store (score, frame) tuples
    
    # Loop through each frame in the second
    for frame_num in range(start_frame, start_frame + fps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            sharpness_score = calculate_sharpness(frame)  # Calculate sharpness score
            frame_scores.append((sharpness_score, frame, frame_num))
        else:
            print(f"Error reading frame: {frame_num}")

    # Sort frames by sharpness score in descending order
    frame_scores.sort(reverse=True, key=lambda x: x[0])

    # Pick top 5 frames with highest sharpness scores
    top_5_frames = frame_scores[:5]
    
    # Save top 5 frames and record sharpness scores in text file
    for saved_count, (score, frame, frame_num) in enumerate(top_5_frames):
        frame_filename = os.path.join(frames_folder, f"frame_{sec + 1}_{saved_count + 1}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename} with sharpness score {score:.2f}")

        # Save the sharpness score to the text file
        with open(clarity_file, "a") as f:
            f.write(f"{sec + 1}\tframe_{sec + 1}_{saved_count + 1}.jpg\t{score:.2f}\n")

cap.release()
print("Frame extraction and saving completed.")
print(f"All images saved in: {frames_folder}")
print(f"Clarity scores saved in: {clarity_file}")

subprocess.run(["python", "ROIExtractionScript.py", frames_folder])