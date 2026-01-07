import streamlit as st
import cv2
import numpy as np
import time
import os
import tempfile
import matplotlib.pyplot as plt

st.set_page_config(page_title="VisionLock - Military SOT", layout="centered")
st.title("🎯 VisionLock - Military SOT System")
st.markdown("🤖 Single Object Tracking System - Military Grade")
st.write("Upload a video, provide initial bounding box, and get the tracked output.")

# ===================== STYLING =====================
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
    url("https://policy-wire.com/wp-content/uploads/2025/05/Pakistan-Day-Parade.jpg");
    background-size: cover;
    background-attachment: fixed;
    color: white;
}
h1 { color: #FFD700; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: rgba(0, 0.7, 0, 0.6);
    color: white;
}
[data-testid="stSidebar"] h1, h2, h3 { color: #00BFFF; }
::-webkit-scrollbar-thumb { background: #FFD700; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar.expander("📌 Project Intro"):
    st.markdown("""
    - Perform **single-object military target tracking** in video streams  
    - Initialize tracking using a **manual bounding box (ROI)**  
    - Apply **adaptive template matching** for robust frame-to-frame tracking  
    - Visualize **target lock indicators** (bounding box, crosshair, aim circle)  
    - Export the **processed tracking video** for analysis or portfolio use  
    """)

with st.sidebar.expander("👨‍💻 Developers Name-ID"):
    st.markdown("""
    - **Rayyan Ahmed: 22F-BSAI-11**
    - **Agha Harris: 22F-BSAI-27** 
    - **Irtat Mobin: 22F-BSAI-29**  
    - **Omaid Ejaz: 22F-BSAI-45**  
    - **Wajhi Qureshi: 22F-BSAI-50**
    """)

with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("""
- 🎯 **OpenCV (Template Matching)** → Core object tracking using adaptive correlation methods  
- 🖼️ **OpenCV Video I/O** → Frame decoding, drawing overlays, MP4 encoding  
- ⚙️ **NumPy** → Pixel-level operations and array manipulation  
- 🌐 **Streamlit** → Interactive UI for video upload, ROI input, and results display  
- 🧪 **Python Standard Libraries** → Time measurement, file handling, temporary storage  
""")


# ===================== UPLOAD =====================
uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

# ===================== PREVIEW =====================
if uploaded_video:
    temp_vid = tempfile.NamedTemporaryFile(delete=False)
    temp_vid.write(uploaded_video.read())
    temp_vid.close()

    cap_preview = cv2.VideoCapture(temp_vid.name)
    ret, frame0 = cap_preview.read()
    cap_preview.release()

    if ret:
        frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(frame0_rgb, origin="upper")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

# ===================== ROI INPUT =====================
st.subheader("Initial Bounding Box (pixels)")
x = st.number_input("x", min_value=0, value=100)
y = st.number_input("y", min_value=0, value=100)
w = st.number_input("width", min_value=10, value=150)
h = st.number_input("height", min_value=10, value=150)

user_filename = st.text_input("Output filename:", value="tracked_video")
start_btn = st.button("🚀 Start Tracking")

# ===================== TRACKER =====================
def run_tracker(video_path, bbox, video_out_path):
    search_expansion = 80
    confidence_thr = 0.55

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        video_out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W, H)
    )

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read video")

    x, y, w, h = bbox
    template = cv2.cvtColor(first_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    method = cv2.TM_CCOEFF_NORMED

    # ===== Metrics =====
    confidence_sum = 0.0
    valid_frames = 0
    lost_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x1 = max(x - search_expansion, 0)
        y1 = max(y - search_expansion, 0)
        x2 = min(x + w + search_expansion, W)
        y2 = min(y + h + search_expansion, H)

        search = gray[y1:y2, x1:x2]
        res = cv2.matchTemplate(search, template, method)
        _, confidence, _, best_loc = cv2.minMaxLoc(res)

        if confidence >= confidence_thr:
            x = x1 + best_loc[0]
            y = y1 + best_loc[1]
            confidence_sum += confidence
            valid_frames += 1
        else:
            lost_frames += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
        cx, cy = x + w//2, y + h//2
        cv2.circle(frame, (cx, cy), w//5, (0,0,255), 2)

        out.write(frame)

    cap.release()
    out.release()

    avg_conf = confidence_sum / max(valid_frames, 1)
    success_rate = (valid_frames / max(total_frames, 1)) * 100

    return video_out_path, avg_conf, lost_frames, success_rate

# ===================== EXECUTION =====================
if uploaded_video and start_btn:
    filename = user_filename.strip()
    if not filename.endswith(".mp4"):
        filename += ".mp4"

    out_path = os.path.join(tempfile.gettempdir(), filename)

    with st.spinner("Processing video..."):
        output_path, avg_conf, lost_frames, success_rate = run_tracker(
            temp_vid.name,
            (x, y, w, h),
            out_path
        )

    st.success("Tracking completed successfully!")

    # ===== VIDEO =====
    st.subheader("🎬 Output Tracked Video")
    with open(output_path, "rb") as f:
        st.video(f.read())

    # ===== METRICS =====
    st.subheader("📊 Tracking Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Confidence", f"{avg_conf:.2f}")
    c2.metric("Target Lost Frames", lost_frames)
    c3.metric("Tracking Success Rate", f"{success_rate:.1f}%")

    # ===== DOWNLOAD =====
    with open(output_path, "rb") as f:
        st.download_button(
            "⬇ Download Output Video",
            data=f,
            file_name=filename,
            mime="video/mp4"
        )


