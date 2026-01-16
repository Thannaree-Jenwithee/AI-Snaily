import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import os
import time

# Page configuration
st.set_page_config(
    page_title="AI-Snaily - YOLOv5 Object Detection",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_yolov5_model(model_path):
    """Load YOLOv5 model with caching"""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    return model

def get_confidence_style(conf):
    """Get visual style based on confidence level"""
    if conf < 0.35:
        return {
            'border_style': 'dashed',
            'border_color': 'red',
            'text_color': 'red',
            'label': f'⚠ Low Conf: {conf:.2f}'
        }
    elif conf < 0.60:
        return {
            'border_style': 'dashed',
            'border_color': 'red',
            'text_color': 'red',
            'label': ''
        }
    elif conf < 0.85:
        return {
            'border_style': 'dotted',
            'border_color': 'yellow',
            'text_color': 'orange',
            'label': ''
        }
    else:
        return {
            'border_style': 'solid',
            'border_color': 'green',
            'text_color': 'green',
            'label': f'✓ High Conf: {conf:.2f}'
        }

def generate_color_map(class_names):
    """Generate unique colors for each class"""
    colors = {}
    color_palette = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (128, 0, 0),      # Maroon
        (0, 128, 0),      # Dark Green
        (0, 0, 128),      # Dark Blue
        (128, 128, 0),    # Olive
        (128, 0, 128),    # Purple
        (0, 128, 128),    # Teal
        (255, 128, 0),    # Orange
        (255, 0, 128),    # Pink
        (128, 255, 0),    # Lime
    ]
    for i, class_name in enumerate(class_names):
        colors[class_name] = color_palette[i % len(color_palette)]
    return colors

def main():
    st.title("🔍 AI-Snaily - YOLOv5 Object Detection")
    st.markdown("YOLOv5 Object Detection Application")
    
    # Check if model exists
    model_path = "weights/v5.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.info("Please ensure 'weights/v5.pt' exists in your project directory")
        st.stop()
    
    # Sidebar for configuration
    st.sidebar.header("Detection Configuration")
    
    # Detection parameters
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.01)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)
    
    # Source selection
    st.sidebar.header("Input Source")
    source_option = st.sidebar.selectbox(
        "Choose Input Source",
        ["Upload Image", "Upload Video", "Webcam"]
    )
    
    try:
        # Load model
        with st.spinner("Loading YOLOv5 model..."):
            model = load_yolov5_model(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
        
        if source_option == "Upload Image":
            handle_image_detection(model, confidence, iou_threshold)
        elif source_option == "Upload Video":
            handle_video_detection(model, confidence, iou_threshold)
        elif source_option == "Webcam":
            handle_webcam_detection(model, confidence, iou_threshold)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Make sure you have installed the required packages: pip install -r requirements-v5.txt")

def handle_image_detection(model, confidence, iou_threshold):
    """Handle image upload and detection"""
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Detection Results")
            
            # Run detection with timing
            with st.spinner("Running detection..."):
                start_time = time.time()
                results = model(image, conf=confidence, iou_threshold=iou_threshold)
                inference_time = time.time() - start_time
            
            # Display results
            annotated_image = results.render()[0]
            st.image(annotated_image, use_container_width=True, channels="BGR")
            
            # Display inference time
            st.info(f"⏱️ Inference Time: {inference_time:.3f} seconds")
            
            # Display detection statistics
            detections = results.xyxy[0]
            if len(detections) > 0:
                st.write(f"**Objects detected:** {len(detections)}")
                
                # Get class names and counts
                class_names = results.names
                class_counts = {}
                
                for det in detections:
                    class_id = int(det[5])
                    class_name = class_names[class_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                st.write("**Detection Summary:**")
                for class_name, count in class_counts.items():
                    st.write(f"- {class_name}: {count}")
                
                # Display confidence distribution
                st.write("**Confidence Levels:**")
                confidences = detections[:, 4].cpu().numpy()
                low_conf = sum(1 for c in confidences if c < 0.35)
                mid_conf = sum(1 for c in confidences if 0.35 <= c < 0.60)
                high_conf = sum(1 for c in confidences if 0.60 <= c < 0.85)
                very_high_conf = sum(1 for c in confidences if c >= 0.85)
                
                if low_conf > 0:
                    st.write(f"- ⚠️ Low (<0.35): {low_conf}")
                if mid_conf > 0:
                    st.write(f"- 🔶 Intermediate (0.35-0.60): {mid_conf}")
                if high_conf > 0:
                    st.write(f"- 🟡 Good (0.60-0.85): {high_conf}")
                if very_high_conf > 0:
                    st.write(f"- ✅ Very High (>0.85): {very_high_conf}")
            else:
                st.write("**No objects detected**")

def handle_video_detection(model, confidence, iou_threshold):
    """Handle video upload and detection"""
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=['mp4', 'avi', 'mov', 'mkv']
    )
    
    if uploaded_video is not None:
        # Save uploaded video
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        st.video("temp_video.mp4")
        
        if st.button("Run Detection on Video"):
            process_video(model, "temp_video.mp4", confidence, iou_threshold)

def handle_webcam_detection(model, confidence, iou_threshold):
    """Handle webcam detection"""
    st.write("### Webcam Detection")
    
    if st.button("Start Webcam Detection"):
        st.info("Webcam detection requires running locally. Use the code below:")
        
        webcam_code = '''
# Run this code locally for webcam detection
import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/v5.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.45)
    annotated_frame = results.render()[0]
    
    cv2.imshow("YOLOv5 Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        '''
        
        st.code(webcam_code, language='python')

def process_video(model, video_path, confidence, iou_threshold):
    """Process video with YOLOv5 detection"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    frame_count = 0
    total_inference_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection with timing
        start_time = time.time()
        results = model(frame, conf=confidence, iou_threshold=iou_threshold)
        frame_inference_time = time.time() - start_time
        total_inference_time += frame_inference_time
        
        annotated_frame = results.render()[0]
        
        # Write frame
        out.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        # Display stats
        avg_inference_time = total_inference_time / frame_count
        stats_placeholder.info(f"⏱️ Avg Inference Time: {avg_inference_time:.3f}s | Frame: {frame_count}/{total_frames}")
        
        # Display current frame (every 30th frame to avoid lag)
        if frame_count % 30 == 0:
            frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
    
    cap.release()
    out.release()
    
    st.success(f"Video processing complete! Total inference time: {total_inference_time:.2f}s")
    st.video("output_video.mp4")

if __name__ == "__main__":
    main()