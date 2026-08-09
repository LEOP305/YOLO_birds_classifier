import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Cockatiel vs Budgerigar Detector", layout="centered")
st.title("🦜 Cockatiel vs Budgerigar Detector")
st.write(
    "Upload an image or video of a bird, and the model will detect whether "
    "it's a **Cockatiel** or a **Budgerigar**."
)

# ----------------------------
# Load model (cached so it only loads once)
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # make sure best.pt is in the same folder as this script

model = load_model()

# ----------------------------
# Sidebar settings
# ----------------------------
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05
)

# ----------------------------
# Mode selection
# ----------------------------
mode = st.radio("Choose input type:", ["Image", "Video"])

# ----------------------------
# IMAGE MODE
# ----------------------------
if mode == "Image":
    uploaded_image = st.file_uploader(
        "Upload a bird image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            with st.spinner("Detecting..."):
                # Ultralytics expects BGR when given a numpy array (like cv2.imread)
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                results = model(img_bgr, conf=conf_threshold)
                annotated_bgr = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                st.image(
                    annotated_rgb,
                    caption="Detection Result",
                    use_column_width=True,
                )

                # Show detection details
                boxes = results[0].boxes
                if len(boxes) == 0:
                    st.warning("No birds detected. Try lowering the confidence threshold.")
                else:
                    st.subheader("Detections")
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        st.write(f"**{cls_name}** — confidence: {conf:.2f}")

# ----------------------------
# VIDEO MODE (bonus)
# ----------------------------
else:
    uploaded_video = st.file_uploader(
        "Upload a bird video", type=["mp4", "mov", "avi"]
    )

    if uploaded_video is not None:
        # Save uploaded video to a temp file so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("Run Detection on Video"):
            with st.spinner("Processing video... this may take a while"):
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 20
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                out_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=conf_threshold, verbose=False)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)

                    frame_idx += 1
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

                cap.release()
                out.release()

            st.success("Done! Here is the annotated video:")
            st.video(out_path)

            with open(out_path, "rb") as f:
                st.download_button(
                    "Download Annotated Video",
                    f,
                    file_name="annotated_output.mp4",
                    mime="video/mp4",
                )

st.markdown("---")
st.caption("Built with Ultralytics YOLOv8 + Streamlit")
