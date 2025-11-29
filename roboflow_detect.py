# ============================================
# Badminton Shuttle Speed Tracker (Roboflow)
# ============================================

from inference import InferencePipeline
import cv2
import numpy as np
import time
import sys

if len(sys.argv) != 2:
    print("Usage: python3 roboflow_detect.py <video_file>")
    sys.exit(1)

video_path = sys.argv[1]


# ----------------------------------------------------------
# 1. COURT SETUP (YOUR VALUES)
# ----------------------------------------------------------

# Pixel coordinates of corners: A -> B -> C -> D
A = (1293, 631)
B = (1904, 701)
C = (1116, 1279)   # off-screen corner
D = (3,   807)

# Real-world double court dimensions (meters)
# A(0,0), B(6.1,0), C(6.1,13.4), D(0,13.4)
dst_pts_m = np.float32([
    [0.0,  0.0],
    [6.1,  0.0],
    [6.1, 13.4],
    [0.0, 13.4]
])

src_pts_px = np.float32([A, B, C, D])

# Homography: pixel -> meters
H_px_to_m = cv2.getPerspectiveTransform(src_pts_px, dst_pts_m)

# --- simple pixel→meter scale along AB for speed calc ---  # <<< added
AB_vec = np.array(B) - np.array(A)                         # <<< added
AB_len_px = float(np.linalg.norm(AB_vec))                  # <<< added
M_PER_PX = 6.1 / AB_len_px if AB_len_px > 0 else 0.0       # <<< added

def project_to_court(x_px, y_px):
    """Project pixel coordinates to real court meters."""
    pt = np.array([[[x_px, y_px]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H_px_to_m)[0][0]
    x_m, y_m = float(mapped[0]), float(mapped[1])

    # Clamp to court boundaries (avoid crazy numbers)
    x_m = max(0.0, min(6.1, x_m))
    y_m = max(0.0, min(13.4, y_m))
    return x_m, y_m


# ----------------------------------------------------------
# 2. SPEED TRACKING STATE
# ----------------------------------------------------------

last_xy_m = None
last_t = None
smooth_speed_kmh = 0.0

MAX_JUMP_M = 10.0       # ignore >10 m frame jumps (still kills crazy outliers)  # <<< changed
MAX_SPEED_MPS = 120.0   # cap 432 km/h
SPEED_ALPHA = 0.35      # EMA smoothing


# ----------------------------------------------------------
# 3. ROBUST DETECTION PARSING (IMPORTANT!)
# ----------------------------------------------------------

def parse_detections(result):
    """Extract shuttle detections regardless of workflow block type."""

    # Newer workflow format
    if "detections_in_image" in result:
        dets = result["detections_in_image"]
        output = []
        for d in dets:
            output.append({
                "x1": d["x1"],
                "y1": d["y1"],
                "x2": d["x2"],
                "y2": d["y2"],
                "conf": d.get("confidence", 0),
            })
        return output

    # If classic Detections() object exists
    preds = result.get("predictions", None)
    if preds is not None and hasattr(preds, "xyxy"):
        xyxy = preds.xyxy
        confs = preds.confidence
        out = []
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            out.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(c)
            })
        return out

    return []


# ----------------------------------------------------------
# 4. SINK FUNCTION (RUNS EVERY FRAME)
# ----------------------------------------------------------

def my_sink(result, video_frame):
    global last_xy_m, last_t, smooth_speed_kmh

    # Annotated image from workflow
    if not result.get("output_image"):
        return

    frame = result["output_image"].numpy_image

    # ---- Extract detections ----
    dets = parse_detections(result)

    if len(dets) == 0:
        # No detections → just display speed
        cv2.putText(frame,
                    f"Speed: {smooth_speed_kmh:.1f} km/h",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 255, 0), 3)
        cv2.imshow("Badminton Tracker", frame)
        cv2.waitKey(1)
        return

    # Use highest confidence detection
    det = max(dets, key=lambda d: d["conf"])
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    conf = det["conf"]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Draw shuttle dot + confidence
    cv2.circle(frame, (int(cx), int(cy)), 8, (0, 0, 255), -1)
    cv2.putText(frame, f"{conf:.2f}", (int(cx)+10, int(cy)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # ---- Project to court coordinates ----
    x_m, y_m = project_to_court(cx, cy)

    # ---- SPEED ----
    # Use frame-based time if available so dt matches video FPS, not wall-clock timing  # <<< changed
    if hasattr(video_frame, "frame_id"):                                               # <<< changed
        now = video_frame.frame_id / 30.0  # 30 = max_fps                              # <<< changed
    else:                                                                            # <<< changed
        now = time.time()                                                            # <<< changed

    if last_xy_m is not None and last_t is not None:
        dt = now - last_t
        if dt > 0:
            # compute distance in pixels, then convert to meters using M_PER_PX       # <<< changed
            dx_px = cx - last_xy_m[0]                                                # <<< changed
            dy_px = cy - last_xy_m[1]                                                # <<< changed
            dist_px = np.sqrt(dx_px*dx_px + dy_px*dy_px)                             # <<< changed
            dist = dist_px * M_PER_PX                                                # <<< changed

            if dist < MAX_JUMP_M:
                speed_mps = dist / dt
                if speed_mps < MAX_SPEED_MPS:
                    # EMA smoothing
                    speed_kmh = speed_mps * 3.6
                    smooth_speed_kmh = (
                        SPEED_ALPHA * speed_kmh +
                        (1-SPEED_ALPHA) * smooth_speed_kmh
                    )

    # store current pixel center as "last_xy_m" for next frame                        # <<< changed
    last_xy_m = (cx, cy)                                                              # <<< changed
    last_t = now

    # ---- Draw speed ----
    cv2.putText(frame,
                f"Speed: {smooth_speed_kmh:.1f} km/h",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (0, 255, 0), 3)

    cv2.imshow("Badminton Tracker", frame)
    cv2.waitKey(1)


# ----------------------------------------------------------
# 5. RUN THE PIPELINE
# ----------------------------------------------------------

pipeline = InferencePipeline.init_with_workflow(
    api_key="3wixWNG4N7Nm9zndSzfF",
    workspace_name="shuttlecock-tracker",
    workflow_id="detect-count-and-visualize",
    video_reference=video_path,
    max_fps=30,
    on_prediction=my_sink
)

pipeline.start()
pipeline.join()
