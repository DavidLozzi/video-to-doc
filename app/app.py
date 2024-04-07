import streamlit as st
import os
import tempfile
import cv2
from docx import Document
from docx.shared import Inches
from skimage.metrics import structural_similarity as ssim

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "8192"
if not os.path.exists("output"):
    os.makedirs("output")
print(os.getcwd())


def is_frame_unique_ssim(current_frame, previous_frame, frame_index):
    if previous_frame is None:
        return True

    # Convert the images to grayscale
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    ssim_value = ssim(current_frame_gray, previous_frame_gray)

    # If SSIM is less than the threshold, the images are considered different
    is_unique = ssim_value < 0.5

    # if is_unique:
    #     frame_path = f"{output_folder}/unique/frame_{frame_index}.jpg"
    #     cv2.imwrite(frame_path, current_frame_gray)
    # else:
    #     frame_path = f"{output_folder}/diff/frame_{frame_index}.jpg"
    #     cv2.imwrite(frame_path, current_frame_gray)

    return is_unique


def extract_unique_frames(uploaded_file):
    print("Extracting unique frames from video...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        file_bytes = uploaded_file.read()
        temp.write(file_bytes)
        temp_file_name = temp.name

    # Open the temporary file with cv2.VideoCapture
    cap = cv2.VideoCapture(temp_file_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames

    previous_frame = None
    frame_index = 0
    prev_unique_frame = False
    images = []

    st.text("Finding frames")
    progress_bar = st.progress(0)
    status = st.empty()

    cols = [st.columns(3) for _ in range(total_frames // 3)]
    col_index = 0
    row_index = 0
    while True:
        print(f"Processing frame {frame_index}...")
        percentage_done = (
            frame_index / total_frames
        ) * 100  # Calculate percentage of frames processed
        progress_bar.progress(int(percentage_done))  # Update progress bar

        ret, frame = cap.read()
        frame_index += 1

        if not ret:
            break

        if frame_index % 5 != 0:
            continue

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        if is_frame_unique_ssim(frame, previous_frame, frame_index):
            print(f"Frame {frame_index} is unique!")
            prev_unique_frame = True
        else:
            print(f"Frame {frame_index} is not unique!")
            if prev_unique_frame:
                print(f"Frame {frame_index} is the last unique frame!")
                prev_unique_frame = False
                frame_path = f"./output/frame_{frame_index}.jpg"
                cv2.imwrite(frame_path, frame)
                images.append(frame_path)
                status.text(f"Found frame {frame_index}")

                cols[row_index][col_index].image(frame_path, width=300)
                col_index += 1
                if col_index == 3:
                    col_index = 0
                    row_index += 1

        previous_frame = frame.copy()

    cap.release()
    return images


def create_doc_with_images(image_paths):
    # Create a new Word document
    doc = Document()

    # Set the margins to 0.5 inches
    for section in doc.sections:
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)

    # Loop over the image paths
    for i, image_path in enumerate(image_paths):
        # Add a new table for each image
        table = doc.add_table(rows=1, cols=2)

        for cell in table.columns[0].cells:
            cell.width = Inches(2.0)
        # Add the image to the first cell
        run = table.cell(0, 0).paragraphs[0].add_run()
        run.add_picture(image_path, width=Inches(2.0))

        # Add text to the second cell
        table.cell(0, 1).text = "."

    # Save the document
    doc.save("./output/output.docx")


def process_file(uploaded_file):
    """Simulate file processing with progress."""

    placeholder_download = st.empty()
    placeholder_status = st.empty()
    images = extract_unique_frames(uploaded_file)
    print(images)
    placeholder_status.text(f"Found {len(images)} frames")
    st.text("Creating Word Doc")
    create_doc_with_images(images)

    st.success("Processing complete!")
    with open("./output/output.docx", "rb") as f:
        data = f.read()

    placeholder_download.download_button(
        label="Download Word Doc",
        data=data,
        file_name="output.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


st.title("Video to Word")
st.subheader(
    "Upload a video to extract unique frames and save them to a Word document."
)

uploaded_file = st.file_uploader("Choose a video")

if uploaded_file is not None:
    if "processed" not in st.session_state or not st.session_state.processed:
        process_file(uploaded_file)
        st.session_state.processed = True
