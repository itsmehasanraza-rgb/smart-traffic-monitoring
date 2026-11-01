
import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from collections import defaultdict
import subprocess

st.set_page_config(
    page_title="Smart Traffic Monitoring",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart Traffic Monitoring System")
st.write("AI-powered Vehicle Detection, Tracking and Counting")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

uploaded_video = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:

    st.success("Video uploaded successfully!")

    st.subheader("Original Video")
    st.video(uploaded_video)

    if st.button("Start Vehicle Detection"):

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp:
            temp.write(uploaded_video.getvalue())
            input_path = temp.name

        output_path = "/tmp/processed_traffic.mp4"

        try:
            model = load_model()

            cap = cv2.VideoCapture(input_path)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            line_y = height // 2

            previous_positions = {}
            counted_ids = set()
            vehicle_counts = defaultdict(int)

            progress = st.progress(0)
            status = st.empty()

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            frame_number = 0

            while cap.isOpened():

                success, frame = cap.read()

                if not success:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[2, 3, 5, 7],
                    conf=0.25,
                    verbose=False
                )

                result = results[0]
                annotated_frame = result.plot()

                cv2.line(
                    annotated_frame,
                    (0, line_y),
                    (width, line_y),
                    (0, 255, 255),
                    3
                )

                if result.boxes is not None:
                    if result.boxes.id is not None:

                        boxes = result.boxes.xyxy.cpu().numpy()
                        ids = result.boxes.id.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()

                        for box, track_id, cls in zip(
                            boxes, ids, classes
                        ):

                            track_id = int(track_id)
                            cls = int(cls)

                            center_y = int(
                                (box[1] + box[3]) / 2
                            )

                            if track_id in previous_positions:

                                previous_y = previous_positions[track_id]

                                crossed = (
                                    (previous_y < line_y and center_y >= line_y)
                                    or
                                    (previous_y > line_y and center_y <= line_y)
                                )

                                if crossed and track_id not in counted_ids:

                                    vehicle_counts[cls] += 1
                                    counted_ids.add(track_id)

                            previous_positions[track_id] = center_y

                cv2.putText(
                    annotated_frame,
                    f"Total Vehicles: {len(counted_ids)}",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                writer.write(annotated_frame)

                frame_number += 1

                if total_frames > 0:
                    progress.progress(
                        min(frame_number / total_frames, 1.0)
                    )

                status.write(
                    f"Processing frame {frame_number}/{total_frames}"
                )

            cap.release()
            writer.release()
            subprocess.run([
                "ffmpeg", "-y",
                "-i", output_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-acodec", "aac",
                "/tmp/processed_traffic_browser.mp4"
            ], check=True)

            browser_video = "/tmp/processed_traffic_browser.mp4"

            st.success("Vehicle detection completed!")

            st.subheader("Vehicle Counting Results")

            names = {
                2: "Cars",
                3: "Motorcycles",
                5: "Buses",
                7: "Trucks"
            }

            total = sum(vehicle_counts.values())

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Total", total)
            col2.metric("Cars", vehicle_counts[2])
            col3.metric("Motorcycles", vehicle_counts[3])
            col4.metric("Buses", vehicle_counts[5])
            col5.metric("Trucks", vehicle_counts[7])

            st.subheader("Processed Traffic Video")

            st.video(browser_video)

            with open(browser_video, "rb") as video_file:
                st.download_button(
                    "Download Processed Video",
                    data=video_file,
                    file_name="processed_traffic.mp4",
                    mime="video/mp4"
                )

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
