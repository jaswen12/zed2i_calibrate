# ZED2i Calibration — Operation Guide

## Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Step 0: Generate ChArUco Board](#step-0-generate-charuco-board)
4. [Step 1: Export Factory Intrinsics](#step-1-export-factory-intrinsics)
5. [Step 2: Stereo Calibration](#step-2-stereo-calibration)
6. [Step 3: Hand-Eye Calibration — Eye-on-Base](#step-3-hand-eye-calibration--eye-on-base)
7. [Step 4: Hand-Eye Calibration — Eye-on-Hand](#step-4-hand-eye-calibration--eye-on-hand)
8. [Step 5: Batch Detection QA](#step-5-batch-detection-qa)
9. [Step 6: Solve Hand-Eye](#step-6-solve-hand-eye)
10. [Step 7: Validate Results](#step-7-validate-results)
11. [Troubleshooting](#troubleshooting)
12. [Appendix: Coordinate Frame Conventions](#appendix-coordinate-frame-conventions)

---

## Overview

This pipeline calibrates the ZED2i stereo camera in two phases:

**Phase 1 — Stereo Intrinsic Calibration** (no robot needed)
- Calibrate focal length, principal point, distortion for left/right cameras
- Calibrate stereo baseline (R, T between left and right)

**Phase 2 — Hand-Eye Calibration** (requires robot)
- Determine the spatial relationship between camera and robot
- Two configurations supported:
  - **Eye-on-base**: camera is fixed externally, board attached to robot flange
  - **Eye-on-hand**: camera mounted on robot flange, board fixed in world

### Data Flow

```
ChArUco board + Camera
        │
        ▼
Step 1: Factory intrinsics ──► zed_intrinsics.yaml
        │
        ▼
Step 2: Stereo calibration ──► stereo_extrinsics.yaml
        │                       (K_left, D_left, K_right, D_right, R, T)
        ▼
Steps 3/4: Collect samples ──► data/eye_on_base/ or data/eye_on_hand/
        │                       (images + robot poses)
        ▼
Step 6: Solve hand-eye ──────► eye_on_base_T.yaml or eye_on_hand_T.yaml
        │                       (4×4 transform matrix)
        ▼
Step 7: Validate ────────────► Quality report
```

---

## Environment Setup

### macOS (Development / Offline Processing)

```bash
cd zed2i_calibrate
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Verify installation
python scripts/run_dry_run.py
```

ZED SDK is not available on macOS. All scripts run in **mock mode** using synthetic data.

### Ubuntu (Real Hardware)

```bash
# 1. Install ZED SDK
# Download from: https://www.stereolabs.com/developers/release
# Follow installer instructions, then:
pip install pyzed  # or use the SDK's Python wheel

# 2. Install this package
cd zed2i_calibrate
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Install HERMES (editable, does NOT modify HERMES source)
pip install -e /path/to/HERMES

# 4. Update config
# Edit config/calibration.yaml:
#   handeye.robot_interface: "flexiv"
#   handeye.robot.serial_number: "your-robot-SN"
#   board specs (after purchasing and measuring)
```

### Configuration

All parameters are in `config/calibration.yaml`. **Never hardcode values in scripts.**

Key settings to update before real calibration:

```yaml
board:
  squares_x: 11              # Match your physical board
  squares_y: 8
  square_length_mm: 30.0     # MEASURE with caliper — printers may scale!
  marker_length_mm: 22.5     # Typically 0.75 × square_length

camera:
  resolution: "HD1080"       # HD2K, HD1080, HD720, VGA
  serial_number: null         # Set if using multiple ZED cameras

handeye:
  mode: "eye_on_base"        # or "eye_on_hand"
  robot_interface: "flexiv"   # "mock" for development
  robot:
    serial_number: "Rizon4s-062626"
```

---

## Step 0: Generate ChArUco Board

Generate a printable board image from config parameters.

```bash
python scripts/00_generate_board.py
python scripts/00_generate_board.py --pps 150   # higher resolution for printing
```

**Output:** `results/charuco_board.png`

**Important:**
- Print the board on a flat, rigid surface (foam board or aluminum composite recommended)
- After printing, **measure the actual square side length with a caliper**
- Update `board.square_length_mm` in `calibration.yaml` if it differs from the expected value
- Even 0.5mm error in square length will degrade calibration accuracy

---

## Step 1: Export Factory Intrinsics

Read the ZED SDK's built-in (factory) intrinsic parameters and save as a baseline.

```bash
python scripts/01_export_zed_intrinsics.py
```

**Output:** `results/zed_intrinsics.yaml`

**What it contains:**
- `K_left`, `D_left` — left camera matrix and distortion
- `K_right`, `D_right` — right camera matrix and distortion
- `R_stereo`, `T_stereo` — factory stereo baseline

These factory values serve as an initial guess for Step 2 and as a fallback if you skip re-calibration.

---

## Step 2: Stereo Calibration

Re-calibrate the ZED2i's intrinsic parameters and stereo baseline using your ChArUco board.

### Live Mode (with camera)

```bash
python scripts/02_stereo_calibrate.py --live
```

**Procedure:**
1. Hold the ChArUco board in front of the camera
2. Preview window shows detected corners overlaid on the image
3. Press **SPACE** to capture when both views show good detection
4. Move the board to a different position/angle and repeat
5. Collect **30-80 samples** (config: `stereo_calibration.min_samples`)
6. Press **Q** to finish and run calibration

**Tips for good samples:**
- Cover the entire field of view (center, corners, edges)
- Vary the board angle (tilt left/right/forward/backward, rotate)
- Vary the distance (close and far)
- Keep the board steady during capture
- Avoid motion blur — lower FPS helps (config: `camera.fps: 15`)
- Ensure even lighting, avoid reflections on the board

### Offline Mode (from saved images)

```bash
python scripts/02_stereo_calibrate.py --offline
python scripts/02_stereo_calibrate.py --offline --samples-dir /path/to/images
```

Expects paired files: `left_0000.png` / `right_0000.png`, `left_0001.png` / `right_0001.png`, ...

**Output:** `results/stereo_extrinsics.yaml`

### Quality Check

- **RMS < 0.5 px**: Excellent
- **RMS 0.5–1.0 px**: Good
- **RMS > 1.0 px**: Consider re-collecting samples with better coverage

---

## Step 3: Hand-Eye Calibration — Eye-on-Base

**Configuration:** Camera is FIXED (tripod/mount), ChArUco board is ATTACHED to robot flange.

```
    [Camera]  ← fixed
        |
        | observes
        ▼
    [Board on Robot Flange]  ← robot moves this
```

### Run

```bash
# With real robot
python scripts/03_collect_eye_on_base.py --live

# With mock robot (development)
python scripts/03_collect_eye_on_base.py --live --mock
```

### Procedure

1. Mount the ChArUco board rigidly on the robot flange
2. Position the camera so it can see the board throughout the robot's workspace
3. Preview window shows:
   - Detected corners + coordinate axes
   - Current TCP pose
   - Sample count
4. Move the robot to a new pose manually (teach pendant or jog mode)
5. Wait for the robot to settle
6. Press **SPACE** — the script waits for settle time, then captures
7. Repeat for **15-50 poses** (config: `handeye.min_samples`)
8. Press **Q** to finish

### Sampling Strategy

For best results, the robot poses should:

- **Span a wide range of rotations** — rotate the board around all 3 axes
- **Include varied translations** — move the board across the camera's FOV
- **Avoid planar motions** — don't just translate in one plane
- **Keep the board visible** — all corners should be detectable
- **Minimum 15 poses**, recommended 25-40

**Output:** `data/eye_on_base/` (samples.json + .npy files)

---

## Step 4: Hand-Eye Calibration — Eye-on-Hand

**Configuration:** Camera is MOUNTED ON the robot flange, ChArUco board is FIXED in the world.

```
    [Camera on Flange]  ← robot moves this
        |
        | observes
        ▼
    [Board]  ← fixed on table/stand
```

### Run

```bash
# With real robot
python scripts/04_collect_eye_on_hand.py --live

# With mock robot (development)
python scripts/04_collect_eye_on_hand.py --live --mock
```

### Procedure

Same interactive workflow as Step 3. The board is now fixed and the robot moves the camera.

**Tips:**
- Secure the board on a stable surface — it must not move during collection
- Ensure the camera can see the board from multiple angles as the robot moves
- The settle time (config: `handeye.settle_time_s`) prevents blur from robot vibration

**Output:** `data/eye_on_hand/` (samples.json + .npy files)

---

## Step 5: Batch Detection QA

Run ChArUco detection on all saved images to check quality before solving.

```bash
python scripts/05_detect_charuco_batch.py data/eye_on_base/images
python scripts/05_detect_charuco_batch.py data/eye_on_hand/images --min-corners 8
python scripts/05_detect_charuco_batch.py data/eye_on_base/images --show  # visualize each
```

**Output:** Per-image report showing corner count, reprojection error, and failure reasons.

Use this to identify bad images to re-capture.

---

## Step 6: Solve Hand-Eye

Solve the hand-eye calibration from collected samples. Runs all 5 OpenCV methods and picks the most consistent one.

```bash
# Use mode from config (default: eye_on_base)
python scripts/06_solve_handeye.py

# Override mode
python scripts/06_solve_handeye.py --mode eye_on_hand

# Single method only
python scripts/06_solve_handeye.py --method TSAI
```

### Available Methods

| Method | Strengths |
|--------|-----------|
| TSAI | Fast, widely used, good baseline |
| PARK | Good for noisy data |
| HORAUD | Robust to outliers |
| ANDREFF | Handles small rotations better |
| DANIILIDIS | Uses dual quaternions, elegant but sensitive to noise |

The script automatically:
1. Runs all 5 methods
2. Performs leave-one-out (LOO) consistency check on each
3. Ranks by combined rotation + translation stability
4. Saves the best result

**Output:**
- `results/eye_on_base_T.yaml` — 4×4 homogeneous transform + metadata
- `results/eye_on_base_T.npy` — NumPy sidecar for quick loading

### Interpreting Results

```
╔══════════════╤═════════════════════════╤═══════════╤═══════════════════════╗
║ Method       │ Translation (mm)        │ Rot (°)   │ LOO Δrot / Δt         ║
╠══════════════╪═════════════════════════╪═══════════╪═══════════════════════╣
║ TSAI         │ x= +12.3 y= -5.1 ...   │   23.45   │ 0.234° / 1.2mm       ║
```

- **Translation**: Camera position relative to robot base (or flange)
- **Rot**: Total rotation angle
- **LOO Δrot / Δt**: Leave-one-out stability — lower = more robust
  - Δrot < 1° and Δt < 5mm → excellent
  - Δrot < 3° and Δt < 15mm → acceptable
  - Higher → consider more/better samples

---

## Step 7: Validate Results

Run quality checks on the solved calibration.

```bash
python scripts/07_validate.py
python scripts/07_validate.py --mode eye_on_hand
```

### What It Checks

**Board-in-base consistency** (primary metric):
- For eye-on-base: projects each board observation through the solved transform to the robot base frame, then checks if the board-to-gripper offset is consistent across all samples
- For eye-on-hand: projects each observation to the base frame, checks if the board's world position is consistent

**Quality thresholds:**
- Std < 5mm → **GOOD**
- Std 5–15mm → **ACCEPTABLE**
- Std > 15mm → **POOR** (consider re-collecting)

### Per-Sample Breakdown

The script shows deviation for each sample. Large outliers indicate:
- Robot was still moving during capture
- Board detection was poor (partial occlusion, blur)
- Robot pose data was stale

You can remove outlier samples from `data/*/samples.json` and re-run Step 6.

---

## Troubleshooting

### "pyzed not found"

Expected on macOS. The system automatically falls back to mock mode. On Ubuntu, install ZED SDK first.

### "Not enough informative motions"

OpenCV's TSAI/ANDREFF methods need sufficient rotation diversity. Solutions:
- Collect more samples with larger rotation differences between poses
- Try other methods (PARK, HORAUD are more tolerant)

### "Rotation normalization issue: determinant(R) is null"

ANDREFF method is sensitive to near-degenerate data. The system automatically skips failed methods and uses others.

### High reprojection error in stereo calibration (> 1.0 px)

- Check that `square_length_mm` matches the physical board
- Ensure even lighting, no reflections
- Remove blurry or partially-occluded samples
- Collect more samples with better board coverage

### Large board-in-base deviation (> 15mm)

- Increase `settle_time_s` in config (robot vibration)
- Check board mounting rigidity
- Verify robot forward kinematics accuracy
- Remove worst samples and re-solve

### "No intrinsics found"

Run Script 01 (export factory intrinsics) before Scripts 03/04/05.

---

## Appendix: Coordinate Frame Conventions

### Transforms Produced

| File | Transform | Meaning |
|------|-----------|---------|
| `stereo_extrinsics.yaml` | R, T | Right camera relative to left camera |
| `eye_on_base_T.yaml` | T_cam←base | Camera frame expressed in robot base frame |
| `eye_on_hand_T.yaml` | T_cam←gripper | Camera frame expressed in gripper (flange) frame |

### Quaternion Convention

HERMES uses **scalar-first** quaternion ordering: `(qw, qx, qy, qz)`

TCP pose from robot: `[x, y, z, qw, qx, qy, qz]` — 7 elements, in metres.

### Using the Results

To transform a point from camera frame to robot base frame (eye-on-base):

```python
import numpy as np
from zed2i_calibrate.io import load_transform

T_cam2base = load_transform("results/eye_on_base_T.yaml", label="T")
T_base_cam = np.linalg.inv(T_cam2base)

# Point in camera frame → robot base frame
p_cam = np.array([x, y, z, 1.0])
p_base = T_base_cam @ p_cam
```

To transform a point from camera frame to robot base frame (eye-on-hand):

```python
T_cam2gripper = load_transform("results/eye_on_hand_T.yaml", label="T")
T_gripper_cam = np.linalg.inv(T_cam2gripper)

# Need the current robot TCP pose
# p_base = T_base_tcp @ T_gripper_cam @ p_cam
```
