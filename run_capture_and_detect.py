import os
import subprocess
import shlex
import sys

BEAGLE_USER_HOST = "kaat@192.168.7.2"

# the actual beagle project path:
REMOTE_PROJECT_DIR = "/home/kaat/ensc351/public/myApps/final_project_camera"

REMOTE_CAPTURE_SCRIPT = "record_15s.py"
REMOTE_VIDEO_PATH = os.path.join(
    REMOTE_PROJECT_DIR, "captured_videos", "capture_15s.mov"
)

LOCAL_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_VIDEO_DIR = os.path.join(LOCAL_PROJECT_DIR, "captured_videos")
LOCAL_VIDEO_PATH = os.path.join(LOCAL_VIDEO_DIR, "capture_15s.mov")


def run_remote_capture():
    cmd = [
        "ssh",
        BEAGLE_USER_HOST,
        f"cd {shlex.quote(REMOTE_PROJECT_DIR)} && "
        f"python3 {shlex.quote(REMOTE_CAPTURE_SCRIPT)}",
    ]
    print("Running remote capture on Beagle:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def copy_video_back():
    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)
    cmd = [
        "scp",
        f"{BEAGLE_USER_HOST}:{REMOTE_VIDEO_PATH}",
        LOCAL_VIDEO_PATH,
    ]
    print("Copying video back to host:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Video copied to:", LOCAL_VIDEO_PATH)


def run_roboflow():
    cmd = [sys.executable, "roboflow_detect.py", LOCAL_VIDEO_PATH]
    print("Running roboflow_detect.py on host:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    run_remote_capture()
    copy_video_back()
    run_roboflow()


if __name__ == "__main__":
    main()
