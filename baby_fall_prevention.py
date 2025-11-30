"""
Baby Fall Prevention System

A state-of-the-art safety pipeline using RT-DETR-Large for person detection
and Depth Anything V2 for depth intelligence to prevent baby falls from beds.

Features:
- RT-DETR-Large detector for high-accuracy person detection
- Depth Anything V2 for real-time depth estimation
- Virtual Depth Wall with configurable Z_threshold
- False positive rejection via aspect ratio analysis
- GPU acceleration with PyTorch
- Depth map heat overlay visualization
"""

import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Represents a single detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    centroid: Tuple[int, int]  # (cx, cy)
    depth: float  # Depth at centroid
    aspect_ratio: float  # height / width
    is_valid_baby: bool
    crossed_depth_wall: bool


class DepthAnythingV2Wrapper:
    """
    Wrapper class for Depth Anything V2 Large model.

    Provides real-time depth estimation from RGB images using the
    Depth Anything V2 architecture for high-quality monocular depth.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the Depth Anything V2 model.

        Args:
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self._load_model()

    def _load_model(self):
        """Load the Depth Anything V2 Large model."""
        try:
            # Try to load from torch hub
            self.model = torch.hub.load(
                "LiheYoung/Depth-Anything",
                "DepthAnything_V2_Large",
                pretrained=True,
                trust_repo=True
            )
        except Exception:
            # Fallback: Try alternative loading method
            try:
                from transformers import pipeline

                self.model = pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Large-hf",
                    device=0 if self.device.type == "cuda" else -1
                )
                self._use_transformers = True
                return
            except ImportError:
                pass

            # If both fail, create a placeholder that returns uniform depth
            print("Warning: Could not load Depth Anything V2. Using fallback depth estimation.")
            self.model = None
            self._use_transformers = False
            return

        self._use_transformers = False
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for depth estimation.

        Args:
            image: BGR image from OpenCV (H, W, 3)

        Returns:
            Preprocessed tensor ready for inference
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size (518x518 for Depth Anything V2 Large)
        resized = cv2.resize(rgb_image, (518, 518))

        # Normalize to [0, 1] and then apply ImageNet normalization
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std

        # Convert to tensor (B, C, H, W)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.float().to(self.device)

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a single RGB image.

        Args:
            image: BGR image from OpenCV (H, W, 3)

        Returns:
            Depth map normalized to [0, 1] where 0 is far and 1 is close
        """
        original_h, original_w = image.shape[:2]

        if self.model is None:
            # Fallback: Return uniform depth
            return np.ones((original_h, original_w), dtype=np.float32) * 0.5

        if hasattr(self, '_use_transformers') and self._use_transformers:
            # Use transformers pipeline
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_image = Image.fromarray(rgb_image)
            result = self.model(pil_image)
            depth_map = np.array(result["depth"])
            depth_map = cv2.resize(depth_map, (original_w, original_h))
            # Normalize to [0, 1]
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            return depth_map.astype(np.float32)

        # Use torch hub model
        with torch.no_grad():
            input_tensor = self.preprocess(image)
            depth = self.model(input_tensor)

        # Post-process depth map
        depth_map = depth.squeeze().cpu().numpy()

        # Resize to original image size
        depth_map = cv2.resize(depth_map, (original_w, original_h))

        # Normalize depth to [0, 1]
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        return depth_map.astype(np.float32)

    def get_depth_at_point(self, depth_map: np.ndarray, point: Tuple[int, int]) -> float:
        """
        Get depth value at a specific point.

        Args:
            depth_map: Normalized depth map
            point: (x, y) coordinates

        Returns:
            Depth value at the point (0 = far, 1 = close)
        """
        x, y = point
        h, w = depth_map.shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        return float(depth_map[y, x])

    def convert_to_metric_depth(
        self,
        normalized_depth: float,
        min_depth: float = 0.5,
        max_depth: float = 5.0
    ) -> float:
        """
        Convert normalized depth to approximate metric depth (meters).

        Args:
            normalized_depth: Depth value in [0, 1]
            min_depth: Minimum depth in meters
            max_depth: Maximum depth in meters

        Returns:
            Approximate metric depth in meters
        """
        # Inverse mapping: higher normalized value = closer = smaller metric depth
        metric_depth = max_depth - normalized_depth * (max_depth - min_depth)
        return metric_depth


class RTDETRDetector:
    """
    RT-DETR-Large detector using Ultralytics implementation.

    Optimized for real-time detection with transformer architecture
    for maximum accuracy in person detection.
    """

    def __init__(self, model_size: str = "rtdetr-l", device: str = "cuda"):
        """
        Initialize RT-DETR detector.

        Args:
            model_size: Model size ('rtdetr-l' for Large)
            device: Device to run inference on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.person_class_id = 0  # COCO class ID for 'person'
        self._load_model(model_size)

    def _load_model(self, model_size: str):
        """Load the RT-DETR model."""
        try:
            from ultralytics import RTDETR

            self.model = RTDETR(f"{model_size}.pt")
            self.model.to(self.device)
        except ImportError:
            print("Warning: Ultralytics not installed. Using fallback.")
            self.model = None
        except Exception as e:
            print(f"Warning: Could not load RT-DETR model: {e}")
            self.model = None

    def detect_persons(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect persons in the image.

        Args:
            image: BGR image from OpenCV
            confidence_threshold: Minimum confidence for detections

        Returns:
            List of detection dictionaries with bbox, confidence, centroid
        """
        detections = []

        if self.model is None:
            return detections

        # Run inference
        results = self.model(image, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                # Filter for person class only
                if cls != self.person_class_id:
                    continue

                if conf < confidence_threshold:
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Calculate centroid
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Calculate aspect ratio (height / width)
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = height / width if width > 0 else 0

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "centroid": (cx, cy),
                    "aspect_ratio": aspect_ratio,
                    "width": width,
                    "height": height
                })

        return detections


class BabyFallPreventionSystem:
    """
    Main pipeline for baby fall prevention.

    Combines RT-DETR detection with Depth Anything V2 depth estimation
    to implement a Virtual Depth Wall safety system.
    """

    # Aspect ratio thresholds for baby detection
    # Babies lying down/crawling typically have lower aspect ratios
    BABY_MIN_ASPECT_RATIO = 0.3
    BABY_MAX_ASPECT_RATIO = 1.5

    # Adults standing typically have higher aspect ratios
    ADULT_MIN_ASPECT_RATIO = 1.5

    def __init__(
        self,
        z_threshold: float = 1.5,
        min_depth: float = 0.5,
        max_depth: float = 5.0,
        confidence_threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize the baby fall prevention system.

        Args:
            z_threshold: Depth threshold for the Virtual Depth Wall (meters)
            min_depth: Minimum depth for metric conversion (meters)
            max_depth: Maximum depth for metric conversion (meters)
            confidence_threshold: Minimum confidence for person detection
            device: Device for GPU acceleration
        """
        self.z_threshold = z_threshold
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold
        self.device = device if torch.cuda.is_available() else "cpu"

        # Initialize models
        print(f"Initializing Baby Fall Prevention System on {self.device}...")
        print("Loading RT-DETR-Large detector...")
        self.detector = RTDETRDetector(model_size="rtdetr-l", device=self.device)

        print("Loading Depth Anything V2 Large model...")
        self.depth_estimator = DepthAnythingV2Wrapper(device=self.device)

        self.alarm_active = False
        print("System initialized successfully.")

    def calibrate_depth_wall(self, depth_map: np.ndarray, bed_edge_y: int) -> float:
        """
        Calibrate the Virtual Depth Wall based on bed edge position.

        Args:
            depth_map: Current depth map
            bed_edge_y: Y-coordinate of the bed edge in the image

        Returns:
            Calibrated depth threshold
        """
        h, w = depth_map.shape
        # Sample depth values along the bed edge
        edge_depths = depth_map[bed_edge_y, :]
        median_depth = np.median(edge_depths)

        # Convert to metric depth
        metric_depth = self.depth_estimator.convert_to_metric_depth(
            median_depth, self.min_depth, self.max_depth
        )

        return metric_depth

    def check_aspect_ratio(self, aspect_ratio: float) -> Tuple[bool, str]:
        """
        Check if the aspect ratio indicates a baby vs adult.

        Args:
            aspect_ratio: Height/width ratio of bounding box

        Returns:
            Tuple of (is_valid_baby, classification)
        """
        if self.BABY_MIN_ASPECT_RATIO <= aspect_ratio <= self.BABY_MAX_ASPECT_RATIO:
            return True, "baby"
        elif aspect_ratio > self.ADULT_MIN_ASPECT_RATIO:
            return False, "adult_standing"
        else:
            return True, "unknown"  # Could be crawling baby, treat as potential baby

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult], bool]:
        """
        Process a single frame through the safety pipeline.

        Args:
            frame: BGR image from camera

        Returns:
            Tuple of (visualized_frame, detection_results, alarm_triggered)
        """
        # Step 1: Generate depth map
        depth_map = self.depth_estimator.estimate_depth(frame)

        # Step 2: Detect persons
        detections = self.detector.detect_persons(frame, self.confidence_threshold)

        # Step 3: Process each detection
        results = []
        alarm_triggered = False

        for det in detections:
            cx, cy = det["centroid"]

            # Get depth at centroid
            normalized_depth = self.depth_estimator.get_depth_at_point(depth_map, (cx, cy))
            metric_depth = self.depth_estimator.convert_to_metric_depth(
                normalized_depth, self.min_depth, self.max_depth
            )

            # Check aspect ratio for false positive rejection
            is_valid_baby, classification = self.check_aspect_ratio(det["aspect_ratio"])

            # Check if person crossed the Virtual Depth Wall
            crossed_depth_wall = metric_depth < self.z_threshold

            # Trigger condition: Baby detected AND crossed depth wall
            trigger_alarm = is_valid_baby and crossed_depth_wall

            if trigger_alarm:
                alarm_triggered = True

            result = DetectionResult(
                bbox=det["bbox"],
                confidence=det["confidence"],
                centroid=det["centroid"],
                depth=metric_depth,
                aspect_ratio=det["aspect_ratio"],
                is_valid_baby=is_valid_baby,
                crossed_depth_wall=crossed_depth_wall
            )
            results.append(result)

        # Step 4: Create visualization
        visualized_frame = self.create_visualization(frame, depth_map, results, alarm_triggered)

        self.alarm_active = alarm_triggered
        return visualized_frame, results, alarm_triggered

    def create_visualization(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        results: List[DetectionResult],
        alarm_triggered: bool
    ) -> np.ndarray:
        """
        Create visualization with depth heat overlay and detection boxes.

        Args:
            frame: Original BGR frame
            depth_map: Normalized depth map
            results: List of detection results
            alarm_triggered: Whether alarm is active

        Returns:
            Visualized frame with overlays
        """
        # Create depth heat map overlay
        depth_colored = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8),
            cv2.COLORMAP_MAGMA
        )

        # Blend depth map with original frame (30% depth, 70% original)
        blended = cv2.addWeighted(frame, 0.7, depth_colored, 0.3, 0)

        # Draw Virtual Depth Wall indicator (horizontal line at threshold depth)
        h, w = frame.shape[:2]
        # Find pixels at threshold depth
        threshold_normalized = 1.0 - (self.z_threshold - self.min_depth) / (self.max_depth - self.min_depth)
        threshold_normalized = max(0, min(1, threshold_normalized))

        # Draw detection boxes and information
        for result in results:
            x1, y1, x2, y2 = result.bbox
            cx, cy = result.centroid

            # Color based on alarm status
            if result.crossed_depth_wall and result.is_valid_baby:
                color = (0, 0, 255)  # Red - ALARM
                label = "ALARM!"
            elif result.crossed_depth_wall:
                color = (0, 165, 255)  # Orange - crossed but not baby
                label = "Adult"
            elif result.is_valid_baby:
                color = (0, 255, 0)  # Green - baby safe zone
                label = "Baby (Safe)"
            else:
                color = (255, 255, 0)  # Cyan - adult
                label = "Adult"

            # Draw bounding box
            cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)

            # Draw centroid
            cv2.circle(blended, (cx, cy), 5, color, -1)

            # Draw info text
            info_text = f"{label} | Z:{result.depth:.2f}m | AR:{result.aspect_ratio:.2f}"
            cv2.putText(
                blended, info_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Draw alarm banner if triggered
        if alarm_triggered:
            cv2.rectangle(blended, (0, 0), (w, 60), (0, 0, 255), -1)
            cv2.putText(
                blended, "!!! ALARM - BABY CROSSING BED EDGE !!!",
                (w // 2 - 250, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )

        # Draw depth threshold indicator
        info_y = h - 30
        cv2.putText(
            blended, f"Depth Wall Z < {self.z_threshold}m | Device: {self.device}",
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        return blended

    def run_on_video(self, video_source: int = 0, output_path: Optional[str] = None):
        """
        Run the pipeline on a video source.

        Args:
            video_source: Video source (0 for webcam, or path to video file)
            output_path: Optional path to save output video
        """
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Starting video processing. Press 'q' to quit.")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            visualized, results, alarm = self.process_frame(frame)

            # Display frame count and FPS info
            frame_count += 1
            cv2.putText(
                visualized, f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            # Show frame
            cv2.imshow("Baby Fall Prevention System", visualized)

            # Write to output if configured
            if writer:
                writer.write(visualized)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Calibrate depth wall at current frame center
                depth_map = self.depth_estimator.estimate_depth(frame)
                self.z_threshold = self.calibrate_depth_wall(depth_map, height // 2)
                print(f"Calibrated depth wall to {self.z_threshold:.2f}m")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    def process_single_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Process a single image.

        Args:
            image_path: Path to input image
            output_path: Optional path to save output image

        Returns:
            Visualized image
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        visualized, results, alarm = self.process_frame(frame)

        if output_path:
            cv2.imwrite(output_path, visualized)
            print(f"Output saved to {output_path}")

        # Print results
        print(f"\nDetection Results for {image_path}:")
        print(f"  Total persons detected: {len(results)}")
        print(f"  Alarm triggered: {alarm}")
        for i, r in enumerate(results):
            print(f"  Person {i + 1}:")
            print(f"    - Depth: {r.depth:.2f}m")
            print(f"    - Aspect Ratio: {r.aspect_ratio:.2f}")
            print(f"    - Is Valid Baby: {r.is_valid_baby}")
            print(f"    - Crossed Depth Wall: {r.crossed_depth_wall}")

        return visualized


def main():
    """Main entry point for the baby fall prevention system."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Baby Fall Prevention System with RT-DETR and Depth Anything V2"
    )
    parser.add_argument(
        "--source", "-s",
        default="0",
        help="Video source (0 for webcam, or path to video/image file)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for processed video/image"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=1.5,
        help="Depth threshold for Virtual Depth Wall in meters (default: 1.5)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Minimum confidence for person detection (default: 0.5)"
    )
    parser.add_argument(
        "--device", "-d",
        default="cuda",
        help="Device for inference (cuda or cpu)"
    )

    args = parser.parse_args()

    # Initialize system
    system = BabyFallPreventionSystem(
        z_threshold=args.threshold,
        confidence_threshold=args.confidence,
        device=args.device
    )

    # Determine if source is an image or video
    source = args.source
    if source.isdigit():
        source = int(source)
        system.run_on_video(source, args.output)
    elif source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        system.process_single_image(source, args.output)
    else:
        system.run_on_video(source, args.output)


if __name__ == "__main__":
    main()
