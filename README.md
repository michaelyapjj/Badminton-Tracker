# Badminton 3D Shuttle Speed & Smash Detection (Roboflow + OpenCV)
This project is a real-time badminton shuttle tracker that estimates 3D speed, height, and detects smashes using a single iPhone video feed.
It applies court homography, net-based height estimation, and motion analysis to compute realistic shuttle motion and identify true smashes vs normal clears/lifts.

It is purpose-built for coaches, players, and performance analysts who want a lightweight, AI-powered speed-tracking system without additional hardware.


## Features

### 1. Real-Time Shuttle Detection (Roboflow Inference Pipeline)
Uses Roboflow’s workflow API to detect the shuttle every frame
Automatically selects the highest-confidence shuttle prediction

### 2. 3D Position Reconstruction
The system reconstructs shuttle motion in 3D using:
Court homography → maps pixel coordinates → real court meters
Net geometry interpolation → estimates real-world shuttle height
Frame-to-frame motion → calculates 3D displacement

### 3. Accurate 3D Speed Calculation
Converts 3D distance traveled per frame into m/s and km/h
Uses exponential smoothing to stabilize noisy detections
Displays real-time speed at the top-right corner

### 4. Smash Detection
A smash is detected only when:
The shuttle is high enough
Speed exceeds threshold
Vertical velocity is strongly downward
Smashes are shown onscreen for 2 seconds:
```
SMASH: 186 km/h
```

### 6. Net-Hit Detection
If a smash meets downward-speed rules and intersects the net plane,
the system displays:
```
SMASH HIT NET
```

## Tech Stack

Python 3.10+
OpenCV (frame display & geometry)
NumPy (math & vector operations)
Roboflow Inference SDK (real-time shuttle detection)

## Input Sources

Supports videos recorded from:
-iPhone 1080p/4K (scaled or unscaled)
-Webcams
-Saved video files (.MOV, .MP4)
-Tracking works from one single camera angle

## Geometry Used

Court Homography
Transforms pixel coordinates → metric space.
Net-Based Height Estimation
Interpolates the top & bottom of the net at each x-pixel to estimate shuttle height using a vertical fraction.

### 3D Speed
Computed as:
```
dx = x2 - x1
dy = y2 - y1
dz = z2 - z1
distance_m = sqrt(dx² + dy² + dz²)
speed_m_s = distance / dt
speed_km_h = speed_m_s × 3.6
```

## File: roboflow_detect.py
The main tracking script includes:
-Court calibration
-Net geometry
-Height estimation
-Smash logic
-Net-hit logic
-Overlay and rendering
-Real-time pipeline integration
