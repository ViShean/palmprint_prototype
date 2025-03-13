# from flask import Blueprint,render_template,request,redirect,url_for,jsonify
# import random
# from datetime import datetime 



# views = Blueprint('views',__name__,static_folder='static')

# # This file is a blueprint that has lots of urls, routes!
# # Each route has a function which is for this is each view's function


# @views.route('/')
# def home():
#     # now = datetime.now()
#     # formatted_date_time = now.strftime("%a, %b %d %I:%M %p")
#     return render_template("main.html")

# @views.route('/collect')
# def collect():
#     return render_template("collection.html")


# @views.route('/identify')
# def validate():
#    return render_template("identification.html")

# from flask import Blueprint, render_template, Response
# import cv2
# import os
# import time
# from datetime import datetime
# import subprocess

# views = Blueprint('views', __name__, static_folder='static')

# # Function to calculate clarity using Laplacian variance
# def calculate_clarity(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray_image, cv2.CV_64F).var()

# # Create folder for images: Images/YYYY-MM-DD_HH-MM-SS
# current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# image_folder = os.path.join("Images", current_timestamp)
# os.makedirs(image_folder, exist_ok=True)

# # Open the default camera (change index if needed)
# cap = None  # Camera not initialized here

# # Initialize video capture and frame generator when requested
# def generate_frames():
#     global cap
#     print("Starting camera...")
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Camera not opened.")
#         return
    
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to read frame.")
#             break
        
#         # Encode frame as JPEG
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# # Index route
# @views.route('/')
# def index():
#     print("Rendering main page.")
#     return render_template('main.html')

# # Video feed route
# @views.route('/video_feed')
# def video_feed():
#     print("Starting video feed.")
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @views.route('/start_camera', methods=['POST'])
# def start_camera():
#     """Start the camera on button click"""
#     global cap
#     print("Start camera button clicked.")
#     if cap is None:
#         cap = cv2.VideoCapture(0)  # Initialize the camera only when requested
#         if not cap.isOpened():
#             print("Error: Camera failed to open.")
#             return "Camera failed to open", 500
        
#     # Call capture_images after starting the camera
#     capture_images()  # This will start capturing images and process them
    
#     return "Camera started and capturing", 200


# # Capture images route
# @views.route('/capture_images')
# def capture_images():
#     """Capture images every 0.2 seconds for 6 seconds"""
#     global cap
#     print("Starting image capture.")
#     frame_counter = 1
#     last_second = 0
#     record_duration = 6  # seconds
#     start_time = time.time()
#     clarity_file = os.path.join(image_folder, "clarity_scores.txt")
    
#     # Check if the camera is properly opened before starting capture
#     if cap is None or not cap.isOpened():
#         print("Error: Camera not initialized correctly.")
#         return "Camera not initialized", 500

#     while time.time() - start_time < record_duration:
#         elapsed_time = time.time() - start_time
#         current_second = int(elapsed_time)

#         if current_second == 0:
#             continue

#         if current_second != last_second:
#             frame_counter = 1
#             last_second = current_second

#         ret, frame = cap.read()
#         if not ret:
#             print(f"Error: Failed to capture frame at second {current_second}.")
#             break

#         print(f"Frame captured at second {current_second}.")
#         clarity_score = calculate_clarity(frame)

#         # Instead of imshow, directly process and save
#         image_filename = os.path.join(image_folder, f"second_{current_second}_{frame_counter}.jpg")
#         cv2.imwrite(image_filename, frame)
#         print(f"Image saved: {image_filename}, Clarity Score: {clarity_score:.2f}")

#         # Save clarity score to the file
#         with open(clarity_file, "a") as f:
#             f.write(f"second_{current_second}_{frame_counter}.jpg\t{clarity_score:.2f}\n")

#         frame_counter += 1

#         # Wait until the next frame
#         time.sleep(0.2)  # Capture every 0.2 seconds

#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"All images saved in: {image_folder}")
#     print(f"Clarity scores saved in: {clarity_file}")

#     # Run ROIExtractionScript.py
#     print("Running ROIExtractionScript.py...")
#     try:
#         subprocess.run(["python", "ROIExtractionScript.py", image_folder], check=True)
#         print("ROIExtractionScript executed successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error running ROIExtractionScript: {e}")

#     return "Image capture completed", 200


# # Collect route
# @views.route('/collect')
# def collect():
#     print("Rendering collection page.")
#     return render_template("collection.html")

# # Identify route
# @views.route('/identify')
# def identify():
#     print("Rendering identification page.")
#     return render_template("identification.html")


# from flask import Blueprint, render_template, Response
# import cv2
# import os
# import time
# from datetime import datetime
# import subprocess

# views = Blueprint('views', __name__, static_folder='static')

# # Function to calculate clarity using Laplacian variance
# def calculate_clarity(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray_image, cv2.CV_64F).var()

# # Create folder for images: Images/YYYY-MM-DD_HH-MM-SS
# current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# image_folder = os.path.join("Images", current_timestamp)
# os.makedirs(image_folder, exist_ok=True)

# # Open the default camera (change index if needed)
# cap = None  # Camera not initialized here

# # Initialize video capture and frame generator when requested
# def generate_frames():
#     global cap
#     print("Starting camera...")
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Camera not opened.")
#         return
    
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to read frame.")
#             break
        
#         # Calculate clarity score and overlay text on frame
#         clarity_score = calculate_clarity(frame)
#         cv2.putText(frame, f"Clarity: {clarity_score:.2f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.putText(frame, f"Time: {int(time.time())}", (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         # Encode frame as JPEG
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# # Index route
# @views.route('/')
# def index():
#     print("Rendering main page.")
#     return render_template('main.html')

# # Video feed route
# @views.route('/video_feed')
# def video_feed():
#     print("Starting video feed.")
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Start camera route
# @views.route('/start_camera', methods=['POST'])
# def start_camera():
#     """Start the camera on button click"""
#     global cap
#     print("Start camera button clicked.")
#     if cap is None:
#         cap = cv2.VideoCapture(0)  # Initialize the camera only when requested
#         if not cap.isOpened():
#             print("Error: Camera failed to open.")
#             return "Camera failed to open", 500
        
#     # Call capture_images after starting the camera
#     capture_images()  # This will start capturing images and process them
    
#     return "Camera started and capturing", 200

# # Capture images route
# @views.route('/capture_images')
# def capture_images():
#     """Capture images every 0.2 seconds for 6 seconds"""
#     global cap
#     print("Starting image capture.")
#     frame_counter = 1
#     last_second = 0
#     record_duration = 6  # seconds
#     start_time = time.time()
#     clarity_file = os.path.join(image_folder, "clarity_scores.txt")
    
#     # Check if the camera is properly opened before starting capture
#     if cap is None or not cap.isOpened():
#         print("Error: Camera not initialized correctly.")
#         return "Camera not initialized", 500

#     while time.time() - start_time < record_duration:
#         elapsed_time = time.time() - start_time
#         current_second = int(elapsed_time)

#         if current_second == 0:
#             continue

#         if current_second != last_second:
#             frame_counter = 1
#             last_second = current_second

#         ret, frame = cap.read()
#         if not ret:
#             print(f"Error: Failed to capture frame at second {current_second}.")
#             break

#         print(f"Frame captured at second {current_second}.")
#         clarity_score = calculate_clarity(frame)

#         # Instead of imshow, directly process and save
#         image_filename = os.path.join(image_folder, f"second_{current_second}_{frame_counter}.jpg")
#         cv2.imwrite(image_filename, frame)
#         print(f"Image saved: {image_filename}, Clarity Score: {clarity_score:.2f}")

#         # Save clarity score to the file
#         with open(clarity_file, "a") as f:
#             f.write(f"second_{current_second}_{frame_counter}.jpg\t{clarity_score:.2f}\n")

#         frame_counter += 1

#         # Wait until the next frame
#         time.sleep(0.2)  # Capture every 0.2 seconds

#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"All images saved in: {image_folder}")
#     print(f"Clarity scores saved in: {clarity_file}")

#     # Run ROIExtractionScript.py
#     print("Running ROIExtractionScript.py...")
#     try:
#         subprocess.run(["python", "ROIExtractionScript.py", image_folder], check=True)
#         print("ROIExtractionScript executed successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error running ROIExtractionScript: {e}")

#     return "Image capture completed", 200

# # Collect route
# @views.route('/collect')
# def collect():
#     print("Rendering collection page.")
#     return render_template("collection.html")

# # Identify route
# @views.route('/identify')
# def identify():
#     print("Rendering identification page.")
#     return render_template("identification.html")


from flask import Blueprint, render_template, Response, jsonify, stream_with_context
import cv2
import os
import time
from datetime import datetime
import subprocess
import io
import sys
from io import StringIO
import threading



class OutputCapture:
    def __init__(self):
        self.buffer = StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def start(self):
        sys.stdout = self.buffer
        sys.stderr = self.buffer

    def stop(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def get_output(self):
        self.buffer.seek(0)
        output = self.buffer.read()
        self.buffer.seek(0)
        self.buffer.truncate(0)
        return output

output_capture = OutputCapture()
output_capture.start()

def capture_terminal_output():
    while True:
        output = output_capture.get_output()
        if output:
            terminal_output.append(output)
        time.sleep(1)  # Adjust the interval as needed

# Start a background thread to capture terminal output
capture_thread = threading.Thread(target=capture_terminal_output, daemon=True)
capture_thread.start()


views = Blueprint('views', __name__, static_folder='static')

# Function to calculate clarity using Laplacian variance
def calculate_clarity(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()



# Redirect stdout to capture print statements
def capture_print_output():
    sys.stdout = io.StringIO()

# Get captured print output
def get_print_output():
    output = sys.stdout.getvalue()
    sys.stdout = sys.__stdout__  # Restore original stdout
    return output

# Append output to the global terminal_output list
def append_output(output):
    global terminal_output
    terminal_output.append(output)

# Run a command and capture its output
def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr


# Create folder for images: Images/YYYY-MM-DD_HH-MM-SS
current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
image_folder = os.path.join("Images", current_timestamp)
os.makedirs(image_folder, exist_ok=True)

# Declare camera as None to not start automatically
cap = None  # Camera not initialized here

# Global variable to store terminal output
terminal_output = []

# Initialize video capture and frame generator when requested
def generate_frames():
    global cap
    if cap is None:  # If the camera is not initialized yet, don't start it automatically
        return

    print("Starting camera...")
    append_output("Starting camera...")  # Capture output

    if not cap.isOpened():
        cap.open(1)  # Open the default camera when generating frames
        if not cap.isOpened():
            print("Error: Camera failed to open.")
            append_output("Error: Camera failed to open.")  # Capture output
            return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Record the start time for the countdown
    start_time = time.time()
    countdown_duration = 5  # 5 seconds countdown

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            append_output("Error: Failed to read frame.")  # Capture output
            break
        
        # Calculate clarity score
        clarity_score = calculate_clarity(frame)

        # Calculate remaining time for the countdown
        elapsed_time = time.time() - start_time
        remaining_time = max(0, countdown_duration - int(elapsed_time))

        # Overlay clarity score and countdown timer on the frame
        cv2.putText(frame, f"Clarity: {clarity_score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {remaining_time}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Stop the loop when the countdown reaches 0
        if remaining_time <= 0:
            break

# Index route
@views.route('/')
def index():
    print("Rendering main page.")
    return render_template('main.html')


@views.route('/start_camera', methods=['POST'])
def start_camera():
    global cap
    print("Start camera button clicked.")
    if cap is None:
        cap = cv2.VideoCapture(1)  # Initialize the camera only when requested
        if not cap.isOpened():
            print("Error: Camera failed to open.")
            return "Camera failed to open", 500
        else:
            print("Camera opened successfully.")
    return "Camera started", 200


@views.route('/video_feed')
def video_feed():
    global cap
    if cap is None:
        print("Error: Camera not initialized.")
        return "Camera not initialized", 500
    if not cap.isOpened():
        print("Error: Camera failed to open.")
        return "Camera failed to open", 500
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Capture images route
@views.route('/capture_images', methods=['POST'])
def capture_images():
    """Capture images every 0.2 seconds for 6 seconds"""
    global cap
    
    print("Starting image capture.")
    
    try:
        frame_counter = 1
        last_second = 0
        record_duration = 6  # seconds
        start_time = time.time()
        clarity_file = os.path.join(image_folder, "clarity_scores.txt")
        
        # Check if the camera is properly opened before starting capture
        if cap is None or not cap.isOpened():
            print("Error: Camera not initialized correctly.")
            return "Camera not initialized", 500

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
                print(f"Error: Failed to capture frame at second {current_second}.")
                break

            print(f"Frame captured at second {current_second}.")
            clarity_score = calculate_clarity(frame)

            # Save the frame as an image
            image_filename = os.path.join(image_folder, f"second_{current_second}_{frame_counter}.jpg")
            cv2.imwrite(image_filename, frame)
            print(f"Image saved: {image_filename}, Clarity Score: {clarity_score:.2f}")

            # Save clarity score to the file
            with open(clarity_file, "a") as f:
                f.write(f"second_{current_second}_{frame_counter}.jpg\t{clarity_score:.2f}\n")

            frame_counter += 1

            # Wait until the next frame
            time.sleep(0.2)  # Capture every 0.2 seconds

        print(f"All images saved in: {image_folder}")
        print(f"Clarity scores saved in: {clarity_file}")
    finally:
           if cap is not None:
            cap.release()
            print("Camera released.")


    # Run ROIExtractionScript.py
    print("Running ROIExtractionScript.py...")
    try:
        subprocess.run(["python", "ROIExtractionScript.py", image_folder], check=True)
        print("ROIExtractionScript executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running ROIExtractionScript: {e}")

    return "Image capture completed", 200

# Collect route
@views.route('/collect')
def collect():
    print("Rendering collection page.")
    return render_template("collection.html")

# Identify route
@views.route('/identify')
def identify():
    print("Rendering identification page.")
    return render_template("identification.html")


@views.route('/stream_terminal_output')
def stream_terminal_output():
    def generate():
        while True:
            if terminal_output:
                yield f"data: {terminal_output[-1]}\n\n"  # Send the latest output
            time.sleep(1)  # Adjust the interval as needed
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@views.route('/get_terminal_output')
def get_terminal_output():
    return jsonify(terminal_output)