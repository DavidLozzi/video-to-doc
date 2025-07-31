#!/usr/bin/env python3
"""
Video Agent - CLI tool for processing videos and generating documentation

This script processes video files, transcripts, and chat logs
to generate comprehensive documentation using AI analysis.
"""

import argparse
import asyncio
import base64
import json
import traceback
import cv2
import httpx
import logging
import os
import re
import subprocess
import sys
import tiktoken
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict, Tuple, Optional


# Configure logging
def setup_logging(log_file: str = "video_agent.log") -> logging.Logger:
    """Setup logging with both file and console handlers"""
    logger = logging.getLogger("video_agent")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler for detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler for status updates
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class VideoAgent:
    """Main class for video processing and documentation generation"""

    def __init__(self, folder_path: str, output_format: str = "md"):
        self.folder_path = Path(folder_path)
        self.output_format = output_format.lower()
        self.logger = setup_logging()

        # Load environment variables
        load_dotenv(override=True)
        self.apim_key = os.getenv("APIM_KEY")
        if not self.apim_key:
            raise ValueError("APIM_KEY not found in environment variables")

        # Video processing parameters
        self.SIMILAR_THRESHOLD = 0.85  # SSIM threshold for frame uniqueness
        self.KEY_FRAMES = 12  # Extract every 12th frame
        self.MAX_TOKENS = 90000

        # Initialize paths
        self.video_path = None
        self.transcript_path = None
        self.chat_path = None
        self.unique_folder = None
        self.all_folder = None

        self.logger.info(f"Initialized VideoAgent for folder: {folder_path}")
        self.logger.info(f"Output format: {output_format}")

    def find_files_by_extension(
        self, extensions: Dict[str, str]
    ) -> Dict[str, Optional[str]]:
        """Find files with specific extensions in the folder"""
        found_files = {ext: None for ext in extensions}

        for ext in extensions:
            for file in self.folder_path.iterdir():
                if file.name.lower().endswith(ext.lower()):
                    found_files[ext] = str(file)
                    break

        return found_files

    def setup_files_and_folders(self) -> bool:
        """Setup and validate required files and folders"""
        self.logger.info("Setting up files and folders...")

        required_extensions = {
            ".mp4": "video_path",
            ".vtt": "transcript_path",
            ".txt": "chat_path",
        }

        files = self.find_files_by_extension(required_extensions.keys())

        self.video_path = files[".mp4"]
        self.transcript_path = files[".vtt"]
        self.chat_path = files[".txt"]

        if not self.video_path or not self.transcript_path:
            self.logger.error(
                f"Required files not found: video={self.video_path}, transcript={self.transcript_path}"
            )
            return False

        self.logger.info(f"Found video: {self.video_path}")
        self.logger.info(f"Found transcript: {self.transcript_path}")

        if self.chat_path:
            self.logger.info(f"Found chat: {self.chat_path}")
        else:
            self.logger.warning("Chat file not found - continuing without it")

        # Create output folders
        self.unique_folder = self.folder_path / "unique"
        self.all_folder = self.folder_path / "all"

        self.unique_folder.mkdir(exist_ok=True)
        self.all_folder.mkdir(exist_ok=True)

        self.logger.info(f"Created folders: {self.unique_folder} and {self.all_folder}")
        return True

    def get_video_stats(self) -> Tuple[float, int]:
        """Get video FPS and total frames"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.logger.info(f"Video stats - FPS: {fps}, Total frames: {total_frames}")
        return fps, total_frames

    def is_frame_unique_ssim(
        self, current_frame, previous_frame, frame_index: int
    ) -> bool:
        """Check if frame is unique using SSIM"""
        if previous_frame is None:
            return True

        # Convert images to grayscale
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between two images
        ssim_value = ssim(current_frame_gray, previous_frame_gray)
        self.logger.debug(
            f"SSIM {frame_index}: {ssim_value*100:.2f} <= {self.SIMILAR_THRESHOLD*100:.2f}"
        )
        is_unique = (ssim_value * 100) < (self.SIMILAR_THRESHOLD * 100)

        return is_unique

    def extract_all_frames(self) -> None:
        """Extract frames from video using FFmpeg"""
        if self.all_folder.exists() and any(self.all_folder.iterdir()):
            self.logger.info(
                f"Frames already extracted in {self.all_folder}. Skipping extraction."
            )
            return

        self.logger.info("Extracting frames from video using FFmpeg...")

        # Use FFmpeg to extract frames at intervals
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            self.video_path,
            "-vf",
            f"select=not(mod(n\\,{self.KEY_FRAMES}))",
            "-vsync",
            "vfr",
            "-frame_pts",
            "true",
            str(self.all_folder / "frame_%05d.jpg"),
        ]

        try:
            subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            self.logger.info("Frame extraction completed successfully")
        except FileNotFoundError:
            self.logger.error("FFmpeg not found. Please install ffmpeg first:")
            self.logger.error("  macOS: brew install ffmpeg")
            self.logger.error("  Linux: sudo apt update && sudo apt install ffmpeg")
            self.logger.error(
                "  Windows: Download from https://ffmpeg.org/download.html"
            )
            raise
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg failed: {e.stderr}")
            raise

    def extract_unique_frames(self) -> List[Dict]:
        """Extract unique frames using SSIM comparison"""
        # Check if unique frames already exist
        if self.unique_folder.exists() and any(self.unique_folder.iterdir()):
            self.logger.info(
                f"Unique frames already exist in {self.unique_folder}. Skipping extraction and processing."
            )
            return self._load_existing_unique_frames()

        self.extract_all_frames()

        self.logger.info("Identifying unique frames...")
        unique_frames = []
        previous_unique_frame = None

        # Get all frame files sorted by number
        frame_files = sorted(
            [f for f in self.all_folder.iterdir() if f.name.startswith("frame_")]
        )
        fps, total_frames = self.get_video_stats()
        cnt = 0
        for frame_index, frame_file in enumerate(frame_files):
            self.logger.debug(f"Processing {frame_index} - {frame_file.name}...")
            cnt += 1
            if cnt % 100 == 0:
                self.logger.info(f"Processed {cnt} frames")

            # Extract frame number from filename (frame_00123.jpg -> 123)
            frame_number_match = re.search(r"frame_(\d+)", frame_file.name)
            if not frame_number_match:
                continue
            frame_number = int(frame_number_match.group(1))

            # Calculate timestamp
            frame_time = frame_number / fps if fps > 0 else 0

            # Load the frame
            frame = cv2.imread(str(frame_file))
            if frame is None:
                self.logger.warning(f"Could not load frame: {frame_file}")
                continue

            # Resize frame for faster comparison
            frame_halfsize = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Check if the frame is unique
            if self.is_frame_unique_ssim(
                frame_halfsize, previous_unique_frame, frame_index
            ):
                self.logger.debug(f"Frame {frame_index} is unique!")
                previous_unique_frame = frame_halfsize.copy()

                # Save to unique frames directory
                output_frame_path = self.unique_folder / frame_file.name
                cv2.imwrite(str(output_frame_path), frame)

                unique_frames.append(
                    {
                        "file_name": frame_file.name,
                        "number": frame_number,
                        "time": frame_time,
                    }
                )
            else:
                self.logger.debug(f"Frame {frame_index} is not unique")

        self.logger.info(f"Extracted {len(unique_frames)} unique frames")
        return unique_frames

    def _load_existing_unique_frames(self) -> List[Dict]:
        """Load existing unique frames from the unique folder"""
        unique_frames = []

        # Get all frame files sorted by number
        frame_files = sorted(
            [f for f in self.unique_folder.iterdir() if f.name.startswith("frame_")]
        )

        fps, _ = self.get_video_stats()

        for frame_file in frame_files:
            # Extract frame number from filename (frame_00123.jpg -> 123)
            frame_number_match = re.search(r"frame_(\d+)", frame_file.name)
            if not frame_number_match:
                continue
            frame_number = int(frame_number_match.group(1))

            # Calculate timestamp
            frame_time = frame_number / fps if fps > 0 else 0

            unique_frames.append(
                {
                    "file_name": frame_file.name,
                    "number": frame_number,
                    "time": frame_time,
                }
            )

        self.logger.info(f"Loaded {len(unique_frames)} existing unique frames")
        return unique_frames

    def convert_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp string to seconds - handles different formats"""
        try:
            # Handle different timestamp formats
            if "." in timestamp:
                # Format: HH:MM:SS.mmm or MM:SS.mmm
                parts = timestamp.split(":")
                if len(parts) == 3:
                    # HH:MM:SS.mmm
                    hours, minutes, seconds_milliseconds = parts
                    seconds, milliseconds = seconds_milliseconds.split(".")
                    total_seconds = (
                        (int(hours) * 3600)
                        + (int(minutes) * 60)
                        + int(seconds)
                        + (int(milliseconds) / 1000)
                    )
                elif len(parts) == 2:
                    # MM:SS.mmm
                    minutes, seconds_milliseconds = parts
                    seconds, milliseconds = seconds_milliseconds.split(".")
                    total_seconds = (
                        (int(minutes) * 60) + int(seconds) + (int(milliseconds) / 1000)
                    )
                else:
                    raise ValueError(f"Unexpected timestamp format: {timestamp}")
            else:
                # Format: HH:MM:SS or MM:SS (no milliseconds)
                parts = timestamp.split(":")
                if len(parts) == 3:
                    # HH:MM:SS
                    hours, minutes, seconds = parts
                    total_seconds = (
                        (int(hours) * 3600) + (int(minutes) * 60) + int(seconds)
                    )
                elif len(parts) == 2:
                    # MM:SS
                    minutes, seconds = parts
                    total_seconds = (int(minutes) * 60) + int(seconds)
                else:
                    raise ValueError(f"Unexpected timestamp format: {timestamp}")

            return total_seconds
        except (ValueError, IndexError) as e:
            self.logger.error(f"Failed to parse timestamp '{timestamp}': {e}")
            return 0.0

    def read_transcript(self) -> List[Dict]:
        """Read and parse VTT transcript file"""
        self.logger.info("Reading transcript file...")

        with open(self.transcript_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        subtitles = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for timestamp lines
            if " --> " in line:
                # Parse start and end times
                start_time, end_time = line.split(" --> ")
                start_seconds = self.convert_timestamp_to_seconds(start_time)
                end_seconds = self.convert_timestamp_to_seconds(end_time)

                # Get text content
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1

                if text_lines:
                    subtitle = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "text": " ".join(text_lines),
                        "speaker": "Unknown",  # VTT doesn't have speaker info
                        "frames": [],
                        "chat": [],
                    }
                    subtitles.append(subtitle)

            i += 1

        self.logger.info(f"Parsed {len(subtitles)} subtitle entries")
        return subtitles

    def parse_chat(self) -> List[Dict]:
        """Parse chat file if available"""
        if not self.chat_path:
            return []

        self.logger.info("Parsing chat file...")

        chat_data = []
        with open(self.chat_path, "r", encoding="utf-8") as file:
            data = {}
            for line in file:
                match = re.match(r"(\d{2}:\d{2}:\d{2})\t(.*?):\t(.*)", line)
                if match:
                    if data:
                        chat_data.append(data)

                    time, name, message = match.groups()
                    data = {"time": time, "name": name, "message": message}
                else:
                    if line.strip() and data:
                        data["message"] += f" {line.strip()}"

            if data:
                chat_data.append(data)

        self.logger.info(f"Parsed {len(chat_data)} chat entries")
        return chat_data

    def sync_frames_with_subtitles(
        self, subtitles: List[Dict], unique_frames: List[Dict], chat_data: List[Dict]
    ) -> List[Dict]:
        """Synchronize frames with subtitles and chat data based on timestamps"""
        self.logger.info("Synchronizing frames with subtitles and chat...")

        synced_subtitles = []

        for subtitle in subtitles:
            # Find frames that fall within this subtitle's time range
            matching_frames = []
            for frame in unique_frames:
                if (
                    subtitle["start_seconds"]
                    <= frame["time"]
                    <= subtitle["end_seconds"]
                ):
                    matching_frames.append(frame)

            # Find chat messages that fall within this subtitle's time range
            matching_chat = []
            for chat in chat_data:
                # Convert chat time to seconds for comparison
                chat_seconds = self.convert_timestamp_to_seconds(chat["time"])
                if subtitle["start_seconds"] <= chat_seconds <= subtitle["end_seconds"]:
                    matching_chat.append(chat)

            # Create synced subtitle entry
            synced_subtitle = subtitle.copy()
            synced_subtitle["frames"] = matching_frames
            synced_subtitle["chat"] = matching_chat

            synced_subtitles.append(synced_subtitle)

        self.logger.info(f"Synchronized {len(synced_subtitles)} subtitle entries")
        return synced_subtitles

    def image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def image_tokens(self, width: int, height: int) -> int:
        """Calculate token cost for image"""
        # OpenAI's image token calculation
        h = (height + 511) // 512
        w = (width + 511) // 512
        return 85 + 170 * h * w

    def system_message(self, content: str) -> Dict:
        """Create system message for API"""
        return {"role": "system", "content": content}

    def developer_message(self, content: str) -> Dict:
        """Create developer message for API"""
        return {"role": "developer", "content": content}

    def user_message(self, content: str) -> Dict:
        """Create user message for API"""
        return {"role": "user", "content": content}

    def image_message(self, text: str, base64_images: List[str]) -> Dict:
        """Create message with text and images"""
        content = [{"type": "text", "text": text}]

        for base64_image in base64_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low",
                    },
                }
            )

        return {"role": "user", "content": content}

    async def call_generative(self, messages: List[Dict]) -> Dict:
        """Call the generative AI API"""
        self.logger.debug("Calling generative AI API...")

        url = "https://bcg-kd-voyager-ingestion-apim.azure-api.net/clone/openai/deployments/gpt-4.1/chat/completions?api-version=2024-02-15-preview"
        headers = {
            "api-key": self.apim_key,
            "Content-Type": "application/json",
        }

        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                self.logger.error(f"API call failed: {e}")
                raise

    async def call_reasoning(
        self, messages, max_retries: int = 3, initial_delay: float = 1.0
    ):
        """Call the reasoning AI API with retry logic"""
        retry_count = 0
        delay = initial_delay

        while retry_count <= max_retries:
            try:
                url = "https://bcg-kd-voyager-ingestion-apim.azure-api.net/clone/openai/deployments/o3-mini/chat/completions?api-version=2024-12-01-preview"
                payload = {"messages": messages}

                headers = {"api-key": self.apim_key, "Content-Type": "application/json"}
                self.logger.debug(f"Calling reasoning API with payload: {payload}")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url, headers=headers, data=json.dumps(payload), timeout=600
                    )
                    if response.status_code == 500:
                        error_response = response.json()
                        if (
                            error_response.get("error", {}).get("message")
                            == "The server had an error while processing your request. Sorry about that!"
                        ):
                            if retry_count < max_retries:
                                retry_count += 1
                                self.logger.warning(
                                    f"Received 500 error, retrying in {delay} seconds (attempt {retry_count}/{max_retries})"
                                )
                                await asyncio.sleep(delay)
                                delay *= 2  # Exponential backoff
                                continue
                            else:
                                self.logger.error("Max retries reached for 500 error")
                                return None

                    if response.status_code != 200:
                        self.logger.error(
                            f"reasoning response code error: {response.status_code}: {response.text}"
                        )
                        return {"error": response.status_code, "message": response.text}

                    response_json = response.json()

                    # Log usage statistics if available
                    if "usage" in response_json:
                        usage = response_json["usage"]
                        self.logger.info(
                            f"API Usage - Prompt: {usage.get('prompt_tokens', 0)}, "
                            f"Completion: {usage.get('completion_tokens', 0)}, "
                            f"Total: {usage.get('total_tokens', 0)}"
                        )
                        if "completion_tokens_details" in usage:
                            reasoning_tokens = usage["completion_tokens_details"].get(
                                "reasoning_tokens", 0
                            )
                            self.logger.info(f"Reasoning tokens: {reasoning_tokens}")

                    self.logger.debug(f"Received response: {response_json}")
                    return response_json
            except (
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
            ) as e:
                if retry_count < max_retries:
                    retry_count += 1
                    self.logger.warning(
                        f"Received timeout error ({type(e).__name__}), retrying in {delay} seconds (attempt {retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
                else:
                    self.logger.error(
                        f"Max retries reached for timeout error: {type(e).__name__}"
                    )
                    return None
            except Exception as e:
                self.logger.error(
                    f"Reasoning Error - Exception type: {type(e).__name__}, Message: '{str(e)}', Args: {e.args}"
                )
                self.logger.error(f"Exception details: {repr(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return None

    async def process_video_content(
        self, synced_subtitles: List[Dict], unique_frames: List[Dict]
    ) -> str:
        """Process video content and generate documentation"""
        self.logger.info("Processing video content with AI...")

        prompt = """You are a GPT, a senior engineer at Boston Consulting Group.

<your_goal>
You will be provided a meeting transcript, chat, and video frames as images from a video where
engineers are discussing various topics during a technical design and sharing session. Your goal
is to generate helpful documentation to be saved in our knowledge base.
</your_goal>

<your_tasks>
1. Review all of the content provided to you.
  a. Each piece of text is from the named speaker, transcribed from the video's audio.
  b. The images are the key frames from the video during this time.
  c. The chat messages are from the chat that occurred during that time.
2. Review the text in relation to the images to understand what is going on at this part of the video.
  a. Extract key text from the images that pertain to what the speaker is talking about.
  b. If you identify a helpful image to include, or an image where a lot of conversation is happening, include it in your output.
    - do not include images of people
3. Use this information to generate documentation.
  a. The audience for this article are engineers, who may or may not have been in this meeting.
  b. The documentation should be detailed, comprehensive, and encompass what the video was about, removing the need to watch the video.
  c. Include any action items, next steps, or decisions that were made.
</your_tasks>

<your_output>
1. Use markdown for formatting.
2. Do not include an intro, summary, conclusion, etc. Just focus on the core topics and conversations.
</your_output>"""

        tik = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0
        messages = []
        gpt_calls = 0
        total_images = 0
        responses = []
        width, height = 0, 0
        temp_subtitles = synced_subtitles.copy()

        self.logger.info(f"Starting with {len(temp_subtitles)} subtitles")

        while temp_subtitles:
            gpt_calls += 1

            while total_tokens < self.MAX_TOKENS and temp_subtitles:
                if len(temp_subtitles[0]["frames"]) + total_images > 50:
                    break

                subtitle = temp_subtitles.pop(0)
                text = "Speakers:\n"
                text += subtitle["text"]

                if subtitle["chat"]:
                    text += "\nChat messages during this time:\n"
                    for chat in subtitle["chat"]:
                        text += f"- {chat['message']}\n"

                text += "Images:\n"
                base64_images = []

                for frame in subtitle["frames"]:
                    frame_path = self.unique_folder / frame["file_name"]
                    if frame_path.exists():
                        base64_image = self.image_to_base64(str(frame_path))
                        base64_images.append(base64_image)

                        if width == 0 or height == 0:
                            with Image.open(frame_path) as img:
                                width, height = img.size
                                self.logger.debug(f"Image size: {width}x{height}")

                        total_tokens += self.image_tokens(width, height)
                        text += f" ![frame_{frame['number']}]({self.unique_folder}/{frame['file_name']})"

                total_tokens += len(tik.encode(text))
                total_images += len(base64_images)
                messages.append(self.image_message(text, base64_images))

            messages.insert(0, self.system_message(prompt))
            total_tokens += len(tik.encode(prompt))

            self.logger.info(
                f"GPT call: {gpt_calls}: tokens {total_tokens} across {len(messages)} messages"
            )

            try:
                response = await self.call_generative(messages)
                responses.append({"messages": messages, "response": response})
            except Exception as e:
                self.logger.error(f"Failed to process batch {gpt_calls}: {e}")
                break

            total_tokens = 0
            total_images = 0
            messages = []

        # Combine all responses
        all_text = ""
        for response in responses:
            content = (
                response.get("response", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            all_text += content + "\n\n**********\n"

        self.logger.info("Generating final consolidated document...")

        # Final consolidation
        final_prompt = """You are a helpful GPT, a senior engineer at BCG.

<your_goal>
You will be provided text that has been summarized from a single technical session across multiple summaries. \
Due to technical constraints, the text has been split into multiple unique summaries for this one meeting. \
Your goal is to combine all of the summaries together into a single coherent \
document that captures all of the important information and create a single output.
</your_goal>

<your_tasks>
1. Review all of the content provided to you.
2. Combine all of the summaries into a single coherent and continuous document. Keep in mind these are all from the same meeting, please group topics accordingly.
3. Use this information to generate helpful documentation. The audience for this article are engineers, who may or may not have been in this meeting.
</your_tasks>

<your_guidelines>
- Include any action items, next steps, or decisions that were made at the start of the document.
- Use images whenever possible to help explain the topics.
- Use markdown for formatting.
  - Do not use horizontal rules or line breaks, rely on headings only.
</your_guidelines>"""

        final_messages = [
            self.developer_message(final_prompt),
            self.user_message(all_text),
        ]
        final_response = await self.call_reasoning(final_messages)

        final_content = (
            final_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        return final_content

    def save_markdown(self, content: str, filename: str = "output.md") -> str:
        """Save content as markdown file"""
        output_path = filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        self.logger.info(f"Saved markdown to: {output_path}")
        return str(output_path)

    def convert_to_pdf(self, md_file: str) -> str:
        """Convert markdown to PDF using pdfkit and wkhtmltopdf"""
        try:
            import markdown
            import pdfkit
            from bs4 import BeautifulSoup
        except ImportError as e:
            self.logger.error(
                f"Required libraries for PDF conversion not installed: {e}"
            )
            self.logger.error(
                "Install with: pip install markdown pdfkit beautifulsoup4"
            )
            self.logger.error("Also install wkhtmltopdf system package")
            raise

        try:
            # Read markdown
            with open(md_file, "r", encoding="utf-8") as f:
                md_content = f.read()

            # Convert to HTML
            html_content = markdown.markdown(md_content)

            # Parse and fix image paths
            soup = BeautifulSoup(html_content, "html.parser")

            for img in soup.find_all("img"):
                src = img["src"]
                if not src.startswith("http"):
                    img["src"] = os.path.abspath(
                        os.path.join(os.path.dirname(md_file), src)
                    )

            # Add CSS for images
            if not soup.head:
                head_tag = soup.new_tag("head")
                soup.insert(0, head_tag)

            style_tag = soup.new_tag("style")
            style_tag.string = """
                img {
                    max-width: 100%;
                    height: auto;
                }
            """
            soup.head.append(style_tag)

            html_content = str(soup)

            # Convert to PDF
            pdf_path = md_file.replace(".md", ".pdf")
            options = {"enable-local-file-access": None}
            pdfkit.from_string(html_content, pdf_path, options=options)

            self.logger.info(f"Converted to PDF: {pdf_path}")
            return pdf_path

        except Exception as e:
            self.logger.error(f"PDF conversion failed: {e}")
            if "wkhtmltopdf" in str(e):
                self.logger.error("wkhtmltopdf not found. Install with:")
                self.logger.error("  macOS: brew install wkhtmltopdf")
                self.logger.error("  Linux: sudo apt install wkhtmltopdf")
            raise

    def convert_to_docx(self, md_file: str) -> str:
        """Convert markdown to DOCX using pypandoc"""
        try:
            import pypandoc
        except ImportError:
            self.logger.error(
                "pypandoc not installed. Install with: pip install pypandoc"
            )
            self.logger.error("Also install pandoc system package:")
            self.logger.error("  macOS: brew install pandoc")
            self.logger.error("  Linux: sudo apt install pandoc")
            raise

        try:
            docx_path = md_file.replace(".md", ".docx")
            pypandoc.convert_file(md_file, "docx", outputfile=docx_path)

            self.logger.info(f"Converted to DOCX: {docx_path}")
            return docx_path

        except Exception as e:
            self.logger.error(f"DOCX conversion failed: {e}")
            if "pandoc" in str(e):
                self.logger.error("pandoc not found. Install with:")
                self.logger.error("  macOS: brew install pandoc")
                self.logger.error("  Linux: sudo apt install pandoc")
            raise

    async def process(self) -> str:
        """Main processing pipeline"""
        self.logger.info("Starting video processing pipeline...")

        # Setup files and folders
        if not self.setup_files_and_folders():
            raise ValueError("Failed to setup required files")

        # Get video stats
        fps, total_frames = self.get_video_stats()

        # Extract unique frames
        unique_images = self.extract_unique_frames()

        # Process transcript
        subtitles = self.read_transcript()

        # Process chat if available
        chat_data = self.parse_chat()

        # Sync frames with subtitles and chat
        synced_subtitles = self.sync_frames_with_subtitles(
            subtitles, unique_images, chat_data
        )

        # Process content with AI
        documentation = await self.process_video_content(
            synced_subtitles, unique_images
        )

        # Save as markdown
        md_file = self.save_markdown(documentation)

        # Convert to other formats if requested
        if self.output_format == "pdf":
            return self.convert_to_pdf(md_file)
        elif self.output_format == "docx":
            return self.convert_to_docx(md_file)
        else:
            return md_file


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Video Agent - Process videos and generate documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./my_video_folder
  %(prog)s ./my_video_folder --format pdf
  %(prog)s ./my_video_folder --format docx
        """,
    )

    parser.add_argument(
        "folder",
        help="Path to folder containing video (.mp4), transcript (.vtt), and optionally chat (.txt) files",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["md", "pdf", "docx"],
        default="docx",
        help="Output format (default: docx)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger("video_agent").setLevel(getattr(logging, args.log_level))

    try:
        agent = VideoAgent(args.folder, args.format)
        output_file = asyncio.run(agent.process())

        print("\n✅ Processing completed successfully!")
        print(f"📄 Output file: {output_file}")

    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
