# import cv2
# import time
# import os
# from datetime import datetime
# import numpy as np

# def calculate_sharpness(image):
#     """Calculate sharpness using Laplacian variance."""
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray_image, cv2.CV_64F).var()

# # --- Video Recording Section ---
# output_folder = "video"
# os.makedirs(output_folder, exist_ok=True)

# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# video_filename = os.path.join(output_folder, f"video_{timestamp}.mp4")

# # Initialize camera with lower resolution for better FPS
# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Reduced from 1920
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Reduced from 1080

# # Get ACTUAL camera FPS
# requested_fps = 30
# actual_fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"Camera's actual FPS: {actual_fps}")

# # Use frame-count based recording instead of time-based
# record_duration = 5  # Seconds
# target_frames = int(record_duration * actual_fps)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(video_filename, fourcc, actual_fps, 
#                      (int(cap.get(3)), int(cap.get(4))))

# frame_count = 0
# start_time = time.time()

# while frame_count < target_frames:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     out.write(frame)
#     cv2.imshow('Video Preview', frame)
#     frame_count += 1

#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Calculate actual performance
# real_duration = time.time() - start_time
# print(f"Requested: {record_duration}s, Actual: {real_duration:.2f}s")
# print(f"Target frames: {target_frames}, Captured: {frame_count}")

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# # --- Frame Extraction Section ---
# video_name = os.path.splitext(os.path.basename(video_filename))[0]
# frames_folder = os.path.join("frames", video_name)
# os.makedirs(frames_folder, exist_ok=True)

# cap = cv2.VideoCapture(video_filename)

# fps = int(cap.get(cv2.CAP_PROP_FPS))
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# duration = total_frames / fps  # Duration in seconds, can be fractional

# print(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration} sec")

# # Open a text file to save clarity scores
# clarity_file = os.path.join(frames_folder, "clarity_scores.txt")
# with open(clarity_file, "w") as f:
#     f.write("Second\tFrame\tSharpness Score\n")  # Write header for the clarity scores

# # Loop through each second of video
# for sec in range(int(duration)):  # Loop through each second
#     start_frame = sec * fps
#     frame_scores = []  # List to store (score, frame) tuples
    
#     # Loop through each frame in the second
#     for frame_num in range(start_frame, start_frame + fps):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#         ret, frame = cap.read()

#         if ret:
#             sharpness_score = calculate_sharpness(frame)  # Calculate sharpness score
#             frame_scores.append((sharpness_score, frame, frame_num))
#         else:
#             print(f"Error reading frame: {frame_num}")

#     # Sort frames by sharpness score in descending order
#     frame_scores.sort(reverse=True, key=lambda x: x[0])

#     # Pick top 5 frames with highest sharpness scores
#     top_5_frames = frame_scores[:5]
    
#     # Save top 5 frames and record sharpness scores in text file
#     for saved_count, (score, frame, frame_num) in enumerate(top_5_frames):
#         frame_filename = os.path.join(frames_folder, f"frame_{sec + 1}_{saved_count + 1}.jpg")
#         cv2.imwrite(frame_filename, frame)
#         print(f"Saved: {frame_filename} with sharpness score {score:.2f}")

#         # Save the sharpness score to the text file
#         with open(clarity_file, "a") as f:
#             f.write(f"{sec + 1}\tframe_{sec + 1}_{saved_count + 1}.jpg\t{score:.2f}\n")

# cap.release()
# print("Frame extraction and saving completed.")
# print(f"Clarity scores saved in: {clarity_file}")


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import cv2
# import time
# import os
# import numpy as np
# from datetime import datetime

# # Function to calculate clarity using Laplacian variance
# def calculate_clarity(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray_image, cv2.CV_64F).var()

# # Create folder: Images/YYYY-MM-DD_HH-MM-SS
# current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# image_folder = os.path.join("Images", current_timestamp)
# os.makedirs(image_folder, exist_ok=True)

# # Open the default camera, change according to the camera index
# cap = cv2.VideoCapture(1)

# # Set video properties (Optional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Reduced resolution for stability
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# # Get actual resolution and FPS
# actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# actual_fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"Camera resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")

# start_time = time.time()
# record_duration = 6  # Seconds
# frame_interval = 0.2  # 5 FPS

# last_second = int(start_time)
# frame_counter = 1

# # Open the text file to save clarity scores
# clarity_file = os.path.join(image_folder, "clarity_scores.txt")
# with open(clarity_file, "w") as f:
#     f.write("Frame\tClarity Score\n")  # Write header for the clarity scores

# while time.time() - start_time < record_duration:
#     elapsed_time = time.time() - start_time
#     current_second = int(elapsed_time)
    
#     if current_second == 0:
#         continue

#     if current_second != last_second:
#         frame_counter = 1
#         last_second = current_second

#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break

#     # Display preview with clarity score
#     clarity_score = calculate_clarity(frame)
#     cv2.putText(frame, f"Clarity: {clarity_score:.2f}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.putText(frame, f"Time: {current_second}s", (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     cv2.imshow('Camera Preview', frame)

#     if frame_counter <= 5:
#         image_filename = os.path.join(image_folder, f"second_{current_second}_{frame_counter}.jpg")
#         cv2.imwrite(image_filename, frame)
        
#         # Print clarity score when the image is saved
#         print(f"Image saved: {image_filename}, Clarity Score: {clarity_score:.2f}")
        
#         # Save the clarity score to the text file
#         with open(clarity_file, "a") as f:
#             f.write(f"second_{current_second}_{frame_counter}.jpg\t{clarity_score:.2f}\n")
        
#         frame_counter += 1

#     # Calculate the time to wait until the next frame
#     next_frame_time = start_time + (current_second - 1 + (frame_counter - 1) * frame_interval)
#     wait_time = max(0, next_frame_time - time.time())
#     time.sleep(wait_time)

#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# print(f"All images saved in: {image_folder}")
# print(f"Clarity scores saved in: {clarity_file}")

