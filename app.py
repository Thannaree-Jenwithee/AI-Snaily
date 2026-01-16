import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time

# Page configuration
st.set_page_config(
    page_title="AI-Snaily - Multi-YOLO Object Detection",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    return YOLO(model_path)

def get_available_models():
    """Get available models from weights folder"""
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        return {}
    
    models = {}
    for file in os.listdir(weights_dir):
        if file.endswith('.pt'):
            model_name = file.replace('.pt', '')
            models[model_name] = os.path.join(weights_dir, file)
    
    # Sort models numerically (v8, v10, v11 instead of v8, v11, v10)
    sorted_models = {}
    def sort_key(item):
        key = item[0]
        # Extract numeric part from model name (e.g., "8" from "v8")
        import re
        match = re.search(r'\d+', key)
        return int(match.group()) if match else float('inf')
    
    for key, value in sorted(models.items(), key=sort_key):
        sorted_models[key] = value
    
    return sorted_models

def get_confidence_style(conf):
    """Get visual style based on confidence level"""
    if conf < 0.35:
        return {
            'color': (0, 0, 255),      # Red in BGR
            'thickness': 2,
            'line_type': 'dashed',
            'label': f'⚠ Low Conf: {conf:.2f}',
            'label_color': (0, 0, 255)  # Red text
        }
    elif conf < 0.60:
        return {
            'color': (0, 0, 255),      # Red in BGR
            'thickness': 2,
            'line_type': 'dashed',
            'label': '',
            'label_color': (0, 0, 255)
        }
    elif conf < 0.85:
        return {
            'color': (0, 255, 255),    # Yellow in BGR
            'thickness': 2,
            'line_type': 'dotted',
            'label': '',
            'label_color': (0, 165, 255)  # Orange text
        }
    else:
        return {
            'color': (0, 255, 0),      # Green in BGR
            'thickness': 2,
            'line_type': 'solid',
            'label': f'✓ High Conf: {conf:.2f}',
            'label_color': (0, 255, 0)  # Green text
        }

def draw_dashed_rectangle(image, pt1, pt2, color, thickness=2):
    """Draw dashed rectangle on image"""
    x1, y1 = pt1
    x2, y2 = pt2
    dash_length = 10
    gap_length = 5
    
    # Top line
    for x in range(x1, x2, dash_length + gap_length):
        end_x = min(x + dash_length, x2)
        cv2.line(image, (x, y1), (end_x, y1), color, thickness)
    
    # Bottom line
    for x in range(x1, x2, dash_length + gap_length):
        end_x = min(x + dash_length, x2)
        cv2.line(image, (x, y2), (end_x, y2), color, thickness)
    
    # Left line
    for y in range(y1, y2, dash_length + gap_length):
        end_y = min(y + dash_length, y2)
        cv2.line(image, (x1, y), (x1, end_y), color, thickness)
    
    # Right line
    for y in range(y1, y2, dash_length + gap_length):
        end_y = min(y + dash_length, y2)
        cv2.line(image, (x2, y), (x2, end_y), color, thickness)

def draw_dotted_rectangle(image, pt1, pt2, color, thickness=2):
    """Draw dotted rectangle on image"""
    x1, y1 = pt1
    x2, y2 = pt2
    dot_spacing = 5
    
    # Top line
    for x in range(x1, x2, dot_spacing):
        cv2.circle(image, (x, y1), thickness, color, -1)
    
    # Bottom line
    for x in range(x1, x2, dot_spacing):
        cv2.circle(image, (x, y2), thickness, color, -1)
    
    # Left line
    for y in range(y1, y2, dot_spacing):
        cv2.circle(image, (x1, y), thickness, color, -1)
    
    # Right line
    for y in range(y1, y2, dot_spacing):
        cv2.circle(image, (x2, y), thickness, color, -1)

def draw_rectangle(image, pt1, pt2, color, thickness=2, line_type='solid'):
    """Draw rectangle with specified line type"""
    if line_type == 'dashed':
        draw_dashed_rectangle(image, pt1, pt2, color, thickness)
    elif line_type == 'dotted':
        draw_dotted_rectangle(image, pt1, pt2, color, thickness)
    else:  # solid
        cv2.rectangle(image, pt1, pt2, color, thickness)

def render_detections_with_custom_style(image, results, model):
    """Render detections with custom confidence-based styling"""
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image_cv = image.copy()
    
    detections = results[0].boxes
    if detections is None or len(detections) == 0:
        return image_cv
    
    # Get detection data
    xyxy = detections.xyxy.cpu().numpy()  # Bounding box coordinates
    conf = detections.conf.cpu().numpy()   # Confidence scores
    cls = detections.cls.cpu().numpy()     # Class IDs
    
    class_names = model.names
    
    # Draw each detection
    for i in range(len(detections)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        confidence = float(conf[i])
        class_id = int(cls[i])
        class_name = str(class_names[class_id])  # Ensure it's a string with proper encoding
        
        # Get style for this confidence level
        style = get_confidence_style(confidence)
        
        # Draw rectangle with appropriate style
        draw_rectangle(image_cv, (x1, y1), (x2, y2), style['color'], style['thickness'], style['line_type'])
        
        # Always draw label with class name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.39  # 30% smaller than 0.56
        font_thickness = 1
        
        # Create label with class name and confidence info
        if style['label']:
            # For high/low confidence with special labels
            label = f"{class_name} {style['label']}"
        else:
            # For intermediate confidence, just show class name and confidence
            label = f"{class_name}: {confidence:.2f}"
        
        # Get text size
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Draw background rectangle for label with some padding
        label_y = y1 - 10 if y1 > 30 else y2 + 25
        bg_rect_pt1 = (x1 - 2, label_y - text_size[1] - 6)
        bg_rect_pt2 = (x1 + text_size[0] + 6, label_y + 6)
        cv2.rectangle(image_cv, bg_rect_pt1, bg_rect_pt2, style['color'], -1)
        
        # Draw text with outline for better contrast
        # Outline
        cv2.putText(image_cv, label, (x1 + 2, label_y), font, font_scale, (0, 0, 0), 3)
        # Main text
        cv2.putText(image_cv, label, (x1 + 2, label_y), font, font_scale, (255, 255, 255), font_thickness)
                   font_scale, (255, 255, 255), font_thickness)
    
    return image_cv

def main():
    st.title("🔍 AI-Snaily - Multi-YOLO Object Detection App")
    st.markdown("Support for YOLOv8, YOLOv10, and YOLOv11 models")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Get available models from weights folder
    available_models = get_available_models()
    
    if not available_models:
        st.error("No models found in 'weights' folder. Please add .pt files to the weights directory.")
        st.stop()
    
    # Model selection from weights folder only
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(available_models.keys()),
        help="Choose from your custom models in the weights folder"
    )
    
    model_path = available_models[selected_model_name]
    st.sidebar.success(f"Selected: {selected_model_name}")
    st.sidebar.info(f"Model path: {model_path}")
    
    # Detection parameters
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)
    
    # Source selection
    st.sidebar.header("Input Source")
    source_option = st.sidebar.selectbox(
        "Choose Input Source",
        ["Upload Image", "Upload Video", "Webcam"]
    )
    
    try:
        # Load model
        with st.spinner("Loading model..."):
            model = load_model(model_path)
        
        if source_option == "Upload Image":
            handle_image_detection(model, confidence, iou_threshold)
        elif source_option == "Upload Video":
            handle_video_detection(model, confidence, iou_threshold)
        elif source_option == "Webcam":
            handle_webcam_detection(model, confidence, iou_threshold)
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

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
                results = model(image, conf=confidence, iou=iou_threshold)
                inference_time = time.time() - start_time
            
            # Display results with custom styling
            annotated_image = render_detections_with_custom_style(image, results, model)
            st.image(annotated_image, use_container_width=True, channels="BGR")
            
            # Display inference time
            st.info(f"⏱️ Inference Time: {inference_time:.3f} seconds")
            
            # Display detection statistics
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                st.write(f"**Objects detected:** {len(detections)}")
                
                # Class counts
                if hasattr(detections, 'cls'):
                    classes = detections.cls.cpu().numpy()
                    class_names = [model.names[int(cls)] for cls in classes]
                    class_counts = {}
                    for name in class_names:
                        class_counts[name] = class_counts.get(name, 0) + 1
                    
                    st.write("**Detection Summary:**")
                    for class_name, count in class_counts.items():
                        st.write(f"- {class_name}: {count}")
                
                # Display confidence distribution
                st.write("**Confidence Levels:**")
                confidences = detections.conf.cpu().numpy()
                low_conf = sum(1 for c in confidences if c < 0.35)
                mid_low_conf = sum(1 for c in confidences if 0.35 <= c < 0.60)
                mid_high_conf = sum(1 for c in confidences if 0.60 <= c < 0.85)
                high_conf = sum(1 for c in confidences if c >= 0.85)
                
                if low_conf > 0:
                    st.write(f"- ⚠️ Very Low (<0.35): {low_conf}")
                if mid_low_conf > 0:
                    st.write(f"- 🔴 Low (0.35-0.60): {mid_low_conf}")
                if mid_high_conf > 0:
                    st.write(f"- 🟡 Intermediate (0.60-0.85): {mid_high_conf}")
                if high_conf > 0:
                    st.write(f"- ✅ High (>0.85): {high_conf}")
        
        # Model Comparison Section
        st.markdown("---")
        st.subheader("📊 Model Comparison")
        
        available_models = get_available_models()
        
        if len(available_models) > 1:
            comparison_data = []
            comparison_cols = st.columns(len(available_models))
            
            for idx, (model_name, model_path) in enumerate(available_models.items()):
                with st.spinner(f"Processing {model_name}..."):
                    model_instance = load_model(model_path)
                    start_time = time.time()
                    results = model_instance(image, conf=confidence, iou=iou_threshold)
                    inference_time = time.time() - start_time
                    
                    detections = results[0].boxes
                    detection_count = len(detections) if detections is not None else 0
                    
                    comparison_data.append({
                        'Model': model_name,
                        'Detections': detection_count,
                        'Inference Time (s)': f"{inference_time:.3f}"
                    })
                    
                    with comparison_cols[idx]:
                        st.write(f"**{model_name}**")
                        # Use custom rendering for comparison
                        annotated_img = render_detections_with_custom_style(image, results, model_instance)
                        st.image(annotated_img, use_container_width=True, channels="BGR")
                        st.metric("Detections", detection_count)
                        st.metric("Inference Time", f"{inference_time:.3f}s")
            
            # Display comparison table
            st.write("### Comparison Summary")
            st.table(comparison_data)

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
from ultralytics import YOLO

model = YOLO("your_model.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.25)
    annotated_frame = results[0].plot()
    
    cv2.imshow("YOLO Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        '''
        
        st.code(webcam_code, language='python')

def process_video(model, video_path, confidence, iou_threshold):
    """Process video with YOLO detection"""
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
        results = model(frame, conf=confidence, iou=iou_threshold)
        frame_inference_time = time.time() - start_time
        total_inference_time += frame_inference_time
        
        # Render with custom styling
        annotated_frame = render_detections_with_custom_style(frame, results, model)
        
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