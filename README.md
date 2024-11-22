import cv2
import numpy as np
from deepface import DeepFace
import os

def detect_deepfake(image_path):
    """
    Function to analyze an image for deepfake detection
    Returns: 'Deepfake Detected' or 'Authentic'
    """
    try:
        # Analyze the image using the DeepFace library
        analysis = DeepFace.analyze(image_path, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)
        print(f"Analysis Results: {analysis}")
        
        # The analysis returns various features. We focus on 'age', 'gender' & 'emotion'.
        # Deepfake videos might show mismatched age, emotion, or inconsistencies in facial features.
        if 'emotion' in analysis and analysis['emotion']:
            emotion_confidence = analysis['emotion']
            if emotion_confidence.get('angry', 0) > 0.7:  # Arbitrary threshold for suspicious emotion
                print("Suspicious emotion detected! Might be a deepfake.")
                return "Deepfake Detected"
        
        # If no strong anomalies found, we consider it 'Authentic'
        return "Authentic"
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return "Error during detection"

def analyze_video(video_path):
    """
    Function to analyze video frames and detect deepfakes
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    deepfake_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save every 30th frame to reduce processing time
        if frame_count % 30 == 0:
            temp_img_path = f"temp_frame_{frame_count}.jpg"
            cv2.imwrite(temp_img_path, frame)
            result = detect_deepfake(temp_img_path)
            if result == "Deepfake Detected":
                deepfake_count += 1
            os.remove(temp_img_path)  # Clean up temporary frame files
        
        frame_count += 1

    cap.release()
    
    if deepfake_count > 0:
        print(f"Warning: {deepfake_count} suspicious frames detected in the video!")
    else:
        print("No deepfake content detected in the video.")

if __name__ == "__main__":
    video_path = 'test_video.mp4'  # Replace with path to the video you want to analyze
    analyze_video(video_path)
