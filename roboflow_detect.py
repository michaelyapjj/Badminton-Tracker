# ============================================
#   Badminton 3D Shuttle Tracker
#   Smash Detection + Net Hit Detection
# ============================================

from inference import InferencePipeline
import cv2
import numpy as np
import time

# ----------------------------------------------------------
# 1. COURT HOMOGRAPHY
# ----------------------------------------------------------

A = (1293, 631)
B = (1904, 701)
C = (1116, 1279)
D = (3,   807)

dst_pts_m = np.float32([
    [0.0,  0.0],
    [6.1,  0.0],
    [6.1, 13.4],
    [0.0, 13.4]
])

src_pts_px = np.float32([A, B, C, D])
H_px_to_m = cv2.getPerspectiveTransform(src_pts_px, dst_pts_m)

def project_to_court(x_px, y_px):
    pt = np.array([[[x_px, y_px]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H_px_to_m)[0][0]
    x = max(0, min(6.1, float(mapped[0])))
    y = max(0, min(13.4, float(mapped[1])))
    return x, y


# ----------------------------------------------------------
# 2. NET HEIGHT ESTIMATION
# ----------------------------------------------------------

NET_TL = (929, 494)
NET_TR = (1797, 546)
NET_BR = (1778, 824)
NET_BL = (924, 683)

NET_HEIGHT_M = 1.55
VERTICAL_GAIN = 1.40  # height correction

def _lerp_y(p0, p1, x):
    x0, y0 = p0; x1, y1 = p1
    if abs(x1 - x0) < 1e-6:
        return (y0 + y1) / 2
    t = (x - x0) / (x1 - x0)
    t = max(0, min(1, t))
    return y0 + t * (y1 - y0)

def net_y_at_x(x):
    min_x = min(NET_TL[0], NET_TR[0], NET_BR[0], NET_BL[0])
    max_x = max(NET_TL[0], NET_TR[0], NET_BR[0], NET_BL[0])
    x = max(min_x, min(max_x, x))
    y_top = _lerp_y(NET_TL, NET_TR, x)
    y_bottom = _lerp_y(NET_BL, NET_BR, x)
    return y_top, y_bottom

def estimate_height(cx, cy):
    y_top, y_bottom = net_y_at_x(cx)
    if y_bottom <= y_top + 1:
        return 0
    frac = (y_bottom - cy) / (y_bottom - y_top)
    frac = max(0, min(2.0, frac))
    return frac * NET_HEIGHT_M * VERTICAL_GAIN


# ----------------------------------------------------------
# 3. SPEED + SMASH STATE
# ----------------------------------------------------------

last_xyz = None
last_t = None
smooth_speed = 0.0

EMA_ALPHA = 0.35
MAX_JUMP = 3.5
MAX_SPEED_MPS = 150

# Smash detection thresholds
smash_active = False
smash_peak_internal = 0.0
last_smash_time = 0

SMASH_MIN_HEIGHT = 2.3
SMASH_MIN_SPEED  = 10.0
SMASH_MIN_DZDT   = -1.2
SMASH_COOLDOWN   = 1.2

DISPLAY_SMASH_GAIN = 12.0
DISPLAY_DURATION = 2.0

overlay_text = ""
overlay_until = 0


# ----------------------------------------------------------
# 4. DETECTION PARSER
# ----------------------------------------------------------

def parse_dets(result):

    if "detections_in_image" in result:
        out = []
        for d in result["detections_in_image"]:
            out.append({
                "x1": d["x1"], "y1": d["y1"],
                "x2": d["x2"], "y2": d["y2"],
                "conf": d.get("confidence", 0)
            })
        return out

    preds = result.get("predictions")
    if preds is not None and hasattr(preds, "xyxy"):
        out = []
        for (x1, y1, x2, y2), c in zip(preds.xyxy, preds.confidence):
            out.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "conf": float(c)
            })
        return out

    return []


# ----------------------------------------------------------
# 5. MAIN PROCESSING + SMASH + NET HIT DETECTION
# ----------------------------------------------------------

def my_sink(result, video_frame):
    global last_xyz, last_t, smooth_speed
    global smash_active, smash_peak_internal, last_smash_time
    global overlay_text, overlay_until

    if not result.get("output_image"):
        return

    frame = result["output_image"].numpy_image
    dets = parse_dets(result)

    if len(dets) == 0:
        # show overlay if still active
        if time.time() < overlay_until:
            cv2.putText(
                frame,
                overlay_text,
                (40, 160),  # moved down so it doesn’t overlap speed/height
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,        # slightly smaller font
                (0, 0, 255),
                4
            )
        cv2.imshow("Tracker", frame)
        cv2.waitKey(1)
        return

    det = max(dets, key=lambda d: d["conf"])
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    cv2.circle(frame, (int(cx), int(cy)), 8, (0, 0, 255), -1)

    x_m, y_m = project_to_court(cx, cy)
    z_m = estimate_height(cx, cy)

    t = time.time()
    dzdt = 0.0

    # SPEED CALCULATION
    if last_xyz is not None and last_t is not None:
        dt = t - last_t
        if dt > 0:
            dx = x_m - last_xyz[0]
            dy = y_m - last_xyz[1]
            dz = z_m - last_xyz[2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            dzdt = dz / dt

            if dist < MAX_JUMP:
                speed_mps = dist / dt
                if speed_mps < MAX_SPEED_MPS:
                    speed_kmh = speed_mps * 3.6
                    smooth_speed = (
                        EMA_ALPHA * speed_kmh +
                        (1 - EMA_ALPHA) * smooth_speed
                    )

    # ------------------------------
    #   SMASH + NET HIT DETECTION
    # ------------------------------
    height_ok = z_m > SMASH_MIN_HEIGHT
    speed_ok  = smooth_speed > SMASH_MIN_SPEED
    down_ok   = dzdt < SMASH_MIN_DZDT

    is_smash_now = height_ok and speed_ok and down_ok

    # Pixel-net detection
    PIXEL_NET_TOL = 35
    NET_HIT_HEIGHT = 1.8
    NET_HIT_DZDT = -0.8

    _, net_bottom_px = net_y_at_x(cx)

    smash_hit_net = False
    if smash_active:
        if (
            abs(cy - net_bottom_px) < PIXEL_NET_TOL and
            z_m < NET_HIT_HEIGHT and
            dzdt < NET_HIT_DZDT
        ):
            smash_hit_net = True

    if is_smash_now:
        if not smash_active:
            smash_active = True
            smash_peak_internal = smooth_speed
        else:
            smash_peak_internal = max(smash_peak_internal, smooth_speed)
    else:
        if smash_active:
            if smash_hit_net:
                overlay_text = "SMASH HIT NET!"
            else:
                final_speed = smash_peak_internal * DISPLAY_SMASH_GAIN
                overlay_text = f"SMASH: {final_speed:.1f} km/h"

            overlay_until = time.time() + DISPLAY_DURATION
            print("\n🔥", overlay_text, "\n")

        smash_active = False
        smash_peak_internal = 0.0

    # update state
    last_xyz = (x_m, y_m, z_m)
    last_t = t

    # --------------------------------------------
    # Speed + height at top-left (non-overlapping)
    # --------------------------------------------
    cv2.putText(
        frame,
        f"Speed: {smooth_speed:.1f} km/h",
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Height: {z_m:.2f} m",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    # smash overlay moved further down
    if time.time() < overlay_until:
        cv2.putText(
            frame,
            overlay_text,
            (40, 160),  # lower than height text
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 0, 255),
            4
        )

    cv2.imshow("Tracker", frame)
    cv2.waitKey(1)


# ----------------------------------------------------------
# 6. START PIPELINE
# ----------------------------------------------------------

pipeline = InferencePipeline.init_with_workflow(
    api_key="3wixWNG4N7Nm9zndSzfF",
    workspace_name="shuttlecock-tracker",
    workflow_id="detect-count-and-visualize",
    video_reference="data/moreSmashes.MOV",
    max_fps=30,
    on_prediction=my_sink
)

pipeline.start()
pipeline.join()
