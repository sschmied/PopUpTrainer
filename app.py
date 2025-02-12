import gradio as gr
import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
import tempfile
import math
import matplotlib.pyplot as plt
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed
engine.setProperty('volume', 1.0)  # Maximum volume

# Ensure YOLOv11-Pose model exists
model_path = "yolo11/yolo11n-pose.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: Model file '{model_path}' not found.")

# Load YOLO model with GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model_pose = YOLO(model_path).to(device)

# Lists to store angles over time
time_stamps = []
back_angles = []
head_angles = []
left_knee_angles = []
right_knee_angles = []

# Repetition tracking variables
position = "down"
repetitions = 0

# Warning timestamps
last_back_warning = 0
last_head_warning = 0
warning_duration = 10  # Persist warnings for 10 seconds

# Function to calculate angle between two vectors
def calculate_angle(a, b, c):
    """Calculate angle at point b given three points (a, b, c)."""
    if a is None or b is None or c is None:
        return None

    ba = np.array(a[:2]) - np.array(b[:2])
    bc = np.array(c[:2]) - np.array(b[:2])

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle if angle >= 0 else 0  # Ensure positive angles

# Function to track repetitions
def track_repetitions(left_knee_angle, right_knee_angle):
    global position, repetitions

    if left_knee_angle > 150 and right_knee_angle > 150:  # DOWN position
        if position == "up":
            repetitions += 1  # Count repetition when returning to down
            engine.say(f"Repetitions: {repetitions}")
            engine.runAndWait()
        position = "down"

    elif left_knee_angle < 100 and right_knee_angle < 100:  # UP position
        position = "up"

# Function to draw skeleton & angles with warnings and repetitions
def draw_skeleton(image, keypoints, frame_time):
    global last_back_warning, last_head_warning
    skeleton_image = np.ones_like(image) * 255  # White background
    connections = [(3, 1), (1, 0), (0, 2), (2, 4), (1, 2), (4, 6), (3, 5), (5, 6), 
                   (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (11, 13), (13, 15), 
                   (12, 14), (14, 16), (5, 11), (6, 12)]
    
    keypoints = keypoints.cpu().numpy()  

    # Get keypoints for angle calculations
    right_shoulder = keypoints[6] if keypoints[6][2] > 0.5 else None
    right_hip = keypoints[12] if keypoints[12][2] > 0.5 else None
    right_eye = keypoints[2] if keypoints[2][2] > 0.5 else None
    right_ear = keypoints[4] if keypoints[4][2] > 0.5 else None
    right_knee = keypoints[14] if keypoints[14][2] > 0.5 else None
    right_ankle = keypoints[16] if keypoints[16][2] > 0.5 else None

    left_hip = keypoints[11] if keypoints[11][2] > 0.5 else None
    left_knee = keypoints[13] if keypoints[13][2] > 0.5 else None
    left_ankle = keypoints[15] if keypoints[15][2] > 0.5 else None
    
    back_angle = calculate_angle(right_hip, right_shoulder, [right_shoulder[0] + 100, right_shoulder[1]])
    head_angle = calculate_angle(right_ear, right_eye, [right_eye[0] + 100, right_eye[1]])
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

    # Track repetitions
    track_repetitions(left_knee_angle, right_knee_angle)

    # Append angles to lists for graphing
    time_stamps.append(frame_time)
    back_angles.append(back_angle if back_angle is not None else 0)
    head_angles.append(head_angle if head_angle is not None else 0)
    left_knee_angles.append(left_knee_angle if left_knee_angle is not None else 0)
    right_knee_angles.append(right_knee_angle if right_knee_angle is not None else 0)

    # Display status and repetitions
    cv2.putText(skeleton_image, f"Status: {position.upper()}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(skeleton_image, f"Repetitions: {repetitions}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return skeleton_image

# Create Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🏄 Pose Estimation App with Repetition Counter")
    gr.Markdown(
        "**A repetition counter. A repetition starts in the down position when both knees are greater than 150 degrees.**"
        " **Halfway is UP when both knees are less than 100 degrees.**"
        " **The repetition is complete when the knees are over 150 degrees again (down).**"
    )

    with gr.Tab("Upload Video"):
        vid_input = gr.File(label="Upload a Video", type="file")
        vid_output = gr.Video(label="Skeleton Video with Angles, Warnings & Repetition Counter")
        vid_button = gr.Button("Process Video")
        vid_button.click(lambda vid: process_video(vid), inputs=vid_input, outputs=vid_output)

    gr.Markdown("Developed for **Hugging Face Spaces** | YOLOv11-Pose + OpenCV | **GPU Supported**")

# Launch Gradio App
demo.launch(share=True)
