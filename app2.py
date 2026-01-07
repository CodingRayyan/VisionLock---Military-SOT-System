import streamlit as st
import cv2
import numpy as np
import time
import os
import tempfile
import matplotlib.pyplot as plt

st.set_page_config(page_title="Template-Based Tank Tracker", layout="centered")
st.title("🎯 VisionLock - Militart SOT System")
st,markdown("Single Object Tracking System - Military Grade")
st.write("Upload a video, provide initial bounding box, and get the tracked output.")

st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                      url("https://policy-wire.com/wp-content/uploads/2025/05/Pakistan-Day-Parade.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}
h1 { color: #FFD700; text-align: center; }
</style>
""", unsafe_allow_html=True)

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

#############################################

uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])


if uploaded_video:
    st.subheader("📐 First Frame Reference (Use this to set Bounding Box)")

    temp_vid = tempfile.NamedTemporaryFile(delete=False)
    temp_vid.write(uploaded_video.read())
    temp_vid.close()

    cap_preview = cv2.VideoCapture(temp_vid.name)
    ret, frame0 = cap_preview.read()
    cap_preview.release()

    if ret:
        frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = frame0_rgb.shape

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(frame0_rgb)

        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_title("First Frame with Pixel Grid")

        ax.set_xticks(np.arange(0, w_img, 50))
        ax.set_yticks(np.arange(0, h_img, 50))
        ax.grid(color="yellow", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.imshow(frame0_rgb, origin="upper")

        st.pyplot(fig)

        st.info(
            "🧭 Tip: Use this grid to estimate **x, y, width, height** values accurately.\n"
            "Coordinates start from **top-left (0,0)** like OpenCV."
        )
    else:
        st.error("Could not read first frame from video.")

st.subheader("Initial Bounding Box (pixels)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    x = st.number_input("x", min_value=0, value=100)
with col2:
    y = st.number_input("y", min_value=0, value=100)
with col3:
    w = st.number_input("width", min_value=10, value=150)
with col4:
    h = st.number_input("height", min_value=10, value=150)

user_filename = st.text_input(
    "Enter output file name (without extension):",
    value="tracked_video"
)

start_btn = st.button("🚀 Start Tracking")

def run_tracker(video_path, bbox, video_out_path):
    search_expansion = 80
    confidence_thr = 0.55
    update_every_n = 10

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

    def choose_method(tmpl):
        mean, std = cv2.meanStdDev(tmpl)
        mean, std = float(mean), float(std)
        if std < 140:
            m = cv2.TM_CCOEFF_NORMED
        else:
            m = cv2.TM_SQDIFF_NORMED
        invert = False
        if mean > 65:
            tmpl[:] = cv2.bitwise_not(tmpl)
            invert = True
        return m, invert

    method, invert_template = choose_method(template)

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x1 = max(x - search_expansion, 0)
        y1 = max(y - search_expansion, 0)
        x2 = min(x + w + search_expansion, W)
        y2 = min(y + h + search_expansion, H)

        search_region = gray[y1:y2, x1:x2]
        if invert_template:
            search_region = cv2.bitwise_not(search_region)

        res = cv2.matchTemplate(search_region, template, method)

        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            confidence = 1.0 - min_val
            best_loc = min_loc
        else:
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            confidence = max_val
            best_loc = max_loc

        if confidence >= confidence_thr:
            search_expansion = 80
            x = x1 + best_loc[0]
            y = y1 + best_loc[1]
        else:
            search_expansion = min(search_expansion + 10, 150)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

        cx, cy = x + w//2, y + h//2
        cross = max(5, w // 8)
        radius = max(8, w // 6)

        cv2.line(frame, (cx-cross, cy), (cx+cross, cy), (0,0,255), 2)
        cv2.line(frame, (cx, cy-cross), (cx, cy+cross), (0,0,255), 2)
        cv2.circle(frame, (cx, cy), radius, (0,0,255), 2)

        cv2.putText(frame, "Military Vehicle Targeted Successfully.", (x, y-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, "FPV Drone AIM Locked.", (x, y+h+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        elapsed = time.time() - start_time
        fps_disp = frame_idx / max(elapsed, 1e-5)
        cv2.putText(frame, f"FPS: {fps_disp:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if frame_idx % update_every_n == 0:
            template = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            method, invert_template = choose_method(template)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return video_out_path

if uploaded_video and start_btn:

    download_name = user_filename.strip()
    if not download_name.endswith(".mp4"):
        download_name += ".mp4"

    video_out_path = os.path.join(tempfile.gettempdir(), download_name)

    with st.spinner("Processing video..."):
        output_path = run_tracker(
            temp_vid.name,
            bbox=(x, y, w, h),
            video_out_path=video_out_path
        )

    st.success(f"Tracking completed successfully! Output file: {download_name}")

    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇ Download Output Video",
            data=f,
            file_name=download_name,
            mime="video/mp4"
        )

    st.code(f"Output saved at:\n{output_path}")


