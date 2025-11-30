# Safetronics Fall Detection V1

## Baby Fall Prevention System

A state-of-the-art Python-based safety pipeline that eliminates false positives caused by perspective distortion and correctly distinguishes toddlers from adults/objects.

### Features

- **RT-DETR-Large Detector**: Uses Ultralytics RT-DETR-Large for maximum accuracy person detection
- **Depth Anything V2 Integration**: Real-time depth estimation using the Large model
- **Virtual Depth Wall**: Configurable depth threshold to detect when someone crosses the bed edge
- **False Positive Rejection**: Aspect ratio analysis to distinguish babies from adults
- **GPU Acceleration**: Full PyTorch GPU support for real-time performance
- **Visualization**: Depth map heat overlay on RGB feed with detection annotations

### Architecture

```
┌─────────────────┐     ┌──────────────────────┐
│   RGB Frame     │────▶│  RT-DETR-Large       │
│   (Camera)      │     │  Person Detection    │
└─────────────────┘     └──────────────────────┘
        │                         │
        │                         ▼
        │               ┌──────────────────────┐
        │               │  Bounding Box +      │
        │               │  Aspect Ratio Check  │
        │               └──────────────────────┘
        │                         │
        ▼                         │
┌─────────────────┐               │
│ Depth Anything  │               │
│ V2 Large        │               │
└─────────────────┘               │
        │                         │
        ▼                         ▼
┌─────────────────┐     ┌──────────────────────┐
│  Depth Map      │────▶│  Virtual Depth Wall  │
│  Generation     │     │  Check: Z < Thresh   │
└─────────────────┘     └──────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────────┐
                        │  ALARM if:           │
                        │  - Person detected   │
                        │  - Is baby (AR check)│
                        │  - Z < Z_threshold   │
                        └──────────────────────┘
```

### Installation

```bash
# Clone the repository
git clone https://github.com/M-Ak21r/Safetronics_fall_detection_V1.git
cd Safetronics_fall_detection_V1

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Command Line Interface

```bash
# Run with webcam (default)
python baby_fall_prevention.py

# Run with video file
python baby_fall_prevention.py --source path/to/video.mp4

# Run with image
python baby_fall_prevention.py --source path/to/image.jpg --output result.jpg

# Customize depth threshold (default: 1.5 meters)
python baby_fall_prevention.py --threshold 2.0

# Run on CPU (if no GPU available)
python baby_fall_prevention.py --device cpu
```

#### Python API

```python
from baby_fall_prevention import BabyFallPreventionSystem

# Initialize the system
system = BabyFallPreventionSystem(
    z_threshold=1.5,       # Virtual Depth Wall at 1.5 meters
    confidence_threshold=0.5,
    device="cuda"          # Use GPU
)

# Process a single frame
import cv2
frame = cv2.imread("test_image.jpg")
visualized, results, alarm_triggered = system.process_frame(frame)

# Check results
for result in results:
    print(f"Depth: {result.depth}m")
    print(f"Is Baby: {result.is_valid_baby}")
    print(f"Crossed Wall: {result.crossed_depth_wall}")

# Run on video
system.run_on_video(video_source=0)  # Webcam
system.run_on_video(video_source="baby_monitor.mp4", output_path="output.mp4")
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `z_threshold` | 1.5 | Depth threshold for Virtual Depth Wall (meters) |
| `min_depth` | 0.5 | Minimum depth for metric conversion |
| `max_depth` | 5.0 | Maximum depth for metric conversion |
| `confidence_threshold` | 0.5 | Minimum confidence for person detection |
| `device` | "cuda" | Device for inference ("cuda" or "cpu") |

### Alarm Trigger Conditions

The alarm is triggered **ONLY** when ALL conditions are met:

1. ✅ A person is detected by RT-DETR-Large
2. ✅ The person's aspect ratio indicates a baby (not an adult standing)
3. ✅ The person's centroid depth Z < Z_threshold (crossed the Virtual Depth Wall)

### False Positive Rejection

The system uses aspect ratio analysis to distinguish:

- **Babies lying/crawling**: Aspect ratio 0.3 - 1.5
- **Adults standing**: Aspect ratio > 1.5

This prevents false alarms from adults walking near the bed.

### Keyboard Controls

- `q`: Quit the application
- `c`: Calibrate depth wall at current frame center

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU with CUDA support (recommended)
- OpenCV 4.8+
- Ultralytics 8.0+

### License

MIT License