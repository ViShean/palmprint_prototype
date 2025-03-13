# import cv2
# import os
# import random

# # Path to the recorded video
# video_filename = "Video/video_2025-02-22_00-39-13.mp4"  # Change to your actual filename

# # Extract filename without extension
# video_name = os.path.splitext(os.path.basename(video_filename))[0]

# # Create a subfolder inside 'frames/' named after the video file
# frames_folder = os.path.join("frames", video_name)
# os.makedirs(frames_folder, exist_ok=True)

# # Open video file
# cap = cv2.VideoCapture(video_filename)

# # Get video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
# duration = total_frames // fps  # Video duration in seconds

# print(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration} sec")

# # Loop through each second in the video
# for sec in range(duration):
#     start_frame = sec * fps  # First frame of the current second
#     frame_indices = random.sample(range(start_frame, start_frame + fps), 10)  # Pick 10 random frames

#     for idx, frame_num in enumerate(frame_indices):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # Move to the frame
#         ret, frame = cap.read()

#         if ret:
#             frame_filename = os.path.join(frames_folder, f"frame_{sec+1}_{idx+1}.jpg")
#             cv2.imwrite(frame_filename, frame)
#             print(f"Saved: {frame_filename}")

# # Release video capture
# cap.release()
# print("Frame extraction completed.")


import cv2
import os
import random

# Path to the recorded video
video_filename = "Video/video_2025-02-22_01-48-07.mp4"  # Change to your actual filename

# Extract filename without extension
video_name = os.path.splitext(os.path.basename(video_filename))[0]

# Create a folder inside 'frames/' named after the video file
frames_folder = os.path.join("frames", video_name)
os.makedirs(frames_folder, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_filename)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
duration = total_frames // fps  # Video duration in seconds

print(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration} sec")

# Loop through each second in the video
for sec in range(duration):
    start_frame = sec * fps  # First frame of the current second
    frame_indices = random.sample(range(start_frame, start_frame + fps), 10)  # Pick 10 random frames

    # Create a subfolder for the current second
    second_folder = os.path.join(frames_folder, f"second_{sec + 1}")
    os.makedirs(second_folder, exist_ok=True)

    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # Move to the frame
        ret, frame = cap.read()

        if ret:
            frame_filename = os.path.join(second_folder, f"frame_{sec + 1}_{idx + 1}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")

# Release video capture
cap.release()
print("Frame extraction completed.")
