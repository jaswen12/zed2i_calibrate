# ZED2i Stereo Camera Calibration Pipeline

Independent calibration system for **ZED2i stereo cameras** paired with **Flexiv Rizon4** robot arms. Performs stereo intrinsic calibration and hand-eye calibration (eye-on-base & eye-on-hand) using ChArUco boards.

Built as a standalone tool, separate from the [HERMES](https://github.com/jaswen12/HERMES) robot control stack.

## Features

- **Stereo calibration** — re-calibrate ZED2i intrinsics + stereo baseline (or verify factory values)
- **Hand-eye calibration** — solve camera-to-robot transform for both mounting configurations:
  - Eye-on-base (camera fixed, board on gripper)
  - Eye-on-hand (camera on flange, board fixed)
- **5 solver methods** — TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS with automatic comparison
- **Leave-one-out validation** — quantitative quality assessment
- **Fully parameterized** — all board specs, camera settings, and robot config in one YAML file
- **Mock mode** — full pipeline runs on macOS without hardware (ZED SDK / robot not required)

## Requirements

- Python 3.9+
- OpenCV with ArUco (opencv-contrib-python >= 4.8)
- NumPy, PyYAML

**For real hardware (Ubuntu only):**
- [ZED SDK](https://www.stereolabs.com/developers/release) + pyzed
- [HERMES](https://github.com/jaswen12/HERMES) (for Flexiv robot bridge)

## Quick Start

```bash
# Clone and install
git clone git@github.com:jaswen12/zed2i_calibrate.git
cd zed2i_calibrate
pip install -e .

# Verify everything works (mock mode, no hardware needed)
python scripts/run_dry_run.py
```

## Pipeline Overview

| Script | Purpose | Hardware Needed |
|--------|---------|-----------------|
| `00_generate_board.py` | Generate printable ChArUco board | None |
| `01_export_zed_intrinsics.py` | Export ZED factory intrinsics | ZED camera |
| `02_stereo_calibrate.py` | Stereo intrinsic calibration | ZED camera + ChArUco board |
| `03_collect_eye_on_base.py` | Collect eye-on-base samples | ZED + robot + board |
| `04_collect_eye_on_hand.py` | Collect eye-on-hand samples | ZED + robot + board |
| `05_detect_charuco_batch.py` | Batch detection quality check | None (uses saved images) |
| `06_solve_handeye.py` | Solve hand-eye calibration | None (uses saved samples) |
| `07_validate.py` | Validate calibration quality | None (uses saved results) |

## Configuration

All parameters in [`config/calibration.yaml`](config/calibration.yaml):

```yaml
board:
  squares_x: 11
  squares_y: 8
  square_length_mm: 30.0      # Measure with caliper after printing!
  marker_length_mm: 22.5
  aruco_dict: "DICT_4X4_100"

camera:
  resolution: "HD1080"

handeye:
  mode: "eye_on_base"          # or "eye_on_hand"
  robot_interface: "mock"      # change to "flexiv" on real hardware
```

## Output

Calibration results are saved as OpenCV YAML files in `results/`:

| File | Contents |
|------|----------|
| `zed_intrinsics.yaml` | Factory K, D for left/right cameras |
| `stereo_extrinsics.yaml` | Calibrated K, D, R, T + RMS error |
| `eye_on_base_T.yaml` | 4x4 camera-to-base transform |
| `eye_on_hand_T.yaml` | 4x4 camera-to-flange transform |

## Project Structure

```
zed2i_calibrate/
├── config/calibration.yaml         # Single source of truth for all parameters
├── src/zed2i_calibrate/
│   ├── board.py                    # ChArUco board utilities
│   ├── camera.py                   # ZED interface (Mock + Real)
│   ├── config.py                   # Config loader
│   ├── handeye_collect.py          # Hand-eye sample collection
│   ├── handeye_solve.py            # Hand-eye solver (5 methods)
│   ├── io.py                       # OpenCV YAML I/O
│   ├── robot_mock.py               # Local mock robot (no HERMES needed)
│   ├── stereo_calibrate.py         # Stereo calibration core
│   └── validate.py                 # Validation & quality assessment
├── scripts/                        # Executable pipeline scripts
├── data/                           # Collected images & poses (gitignored)
└── results/                        # Calibration outputs (gitignored)
```

## License

Internal use.
