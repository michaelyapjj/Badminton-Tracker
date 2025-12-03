import subprocess
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "captured_videos")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "capture_15s.mov")

#logitech c920 camera:
DEVICE = "/dev/video0"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "v4l2",
        "-framerate", "30",
        "-video_size", "1280x720",
        "-i", DEVICE,
        "-t", "15",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        OUTPUT_FILE,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Saved video to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
