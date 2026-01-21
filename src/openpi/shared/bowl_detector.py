"""
Lightweight Bowl Detector using Hough Circle Transform with ROI Constraint

This detector is optimized for speed and accuracy by:
1. Restricting search to a small ROI around the expected bowl position
2. Using Hough Circle Transform which is fast and accurate for circular objects
3. Applying minimal preprocessing (blur + edge detection)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import time


@dataclass
class BowlDetection:
    """Detection result for a bowl."""
    center_x: int
    center_y: int
    radius: int
    confidence: float  # Based on edge strength and circularity


@dataclass
class DetectorConfig:
    """Configuration for the bowl detector."""
    # Expected bowl position (center x, y in full image coordinates)
    # Based on measured data: mean=(326, 406), your spec was (330, 392)
    expected_center: Tuple[int, int] = (326, 406)
    
    # Expected bowl size (measured radius ~47)
    expected_radius: int = 47
    
    # ROI margin around expected center
    roi_margin: int = 60  # Search area: expected_center ± margin
    
    # Hough Circle parameters (tuned for bowl detection)
    min_radius: int = 40
    max_radius: int = 60
    
    # Canny edge detection thresholds
    canny_low: int = 50
    canny_high: int = 150
    
    # Gaussian blur kernel size
    blur_kernel: int = 5
    
    # Hough accumulator threshold (lower = more sensitive)
    hough_param1: int = 100  # Canny high threshold (used internally by HoughCircles)
    hough_param2: int = 25   # Accumulator threshold - slightly lower for reliability
    
    # Minimum distance between detected circle centers
    min_dist: int = 50


class BowlDetector:
    """
    Fast and accurate bowl detector using Hough Circle Transform.
    
    Optimized for detecting a single bowl at a known approximate position.
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._compute_roi_bounds()
    
    def _compute_roi_bounds(self):
        """Precompute ROI bounds for faster processing."""
        cx, cy = self.config.expected_center
        margin = self.config.roi_margin + self.config.expected_radius
        
        self.roi_x1 = max(0, cx - margin)
        self.roi_y1 = max(0, cy - margin)
        self.roi_x2 = cx + margin
        self.roi_y2 = cy + margin
    
    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Extract ROI from frame. Returns (roi, offset_x, offset_y)."""
        h, w = frame.shape[:2]
        
        # Clamp ROI bounds to image dimensions
        x1 = max(0, self.roi_x1)
        y1 = max(0, self.roi_y1)
        x2 = min(w, self.roi_x2)
        y2 = min(h, self.roi_y2)
        
        roi = frame[y1:y2, x1:x2]
        return roi, x1, y1
    
    def detect(self, frame: np.ndarray, sensitive: bool = False) -> Optional[BowlDetection]:
        """
        Detect bowl in the frame.
        
        Args:
            frame: BGR image (640x480)
            sensitive: If True, use more sensitive detection (for empty/low-contrast bowls)
            
        Returns:
            BowlDetection if found, None otherwise
        """
        # Extract ROI for faster processing
        roi, offset_x, offset_y = self._extract_roi(frame)
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.config.blur_kernel, self.config.blur_kernel), 0)
        
        # Apply CLAHE for better contrast (helps with empty bowls)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Adjust threshold based on sensitivity
        param2 = self.config.hough_param2 - 8 if sensitive else self.config.hough_param2
        
        # Try detection on enhanced image first
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,  # Inverse ratio of accumulator resolution
            minDist=self.config.min_dist,
            param1=self.config.hough_param1,
            param2=param2,
            minRadius=self.config.min_radius,
            maxRadius=self.config.max_radius
        )
        
        # Fallback to original blurred if no circles found
        if circles is None and not sensitive:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=self.config.min_dist,
                param1=self.config.hough_param1,
                param2=param2 - 5,  # Lower threshold for fallback
                minRadius=self.config.min_radius,
                maxRadius=self.config.max_radius
            )
        
        if circles is None:
            return None
        
        # Find the best circle (closest to expected center and radius)
        circles = np.around(circles).astype(np.int32)
        best_circle = None
        best_score = float('inf')
        
        expected_cx = self.config.expected_center[0] - offset_x
        expected_cy = self.config.expected_center[1] - offset_y
        expected_r = self.config.expected_radius
        
        for circle in circles[0]:
            cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
            
            # Score based on distance from expected center and radius
            dist_score = np.sqrt((cx - expected_cx)**2 + (cy - expected_cy)**2)
            radius_score = abs(r - expected_r) * 2  # Weight radius difference
            score = dist_score + radius_score
            
            if score < best_score:
                best_score = score
                best_circle = (cx, cy, r)
        
        if best_circle is None:
            return None
        
        cx, cy, r = best_circle
        
        # Compute confidence based on score (lower score = higher confidence)
        # Normalize to 0-1 range
        max_acceptable_score = 50  # Threshold for minimum acceptable detection
        confidence = max(0, 1 - (best_score / max_acceptable_score))
        
        return BowlDetection(
            center_x=int(cx + offset_x),
            center_y=int(cy + offset_y),
            radius=int(r),
            confidence=float(confidence)
        )
    
    def detect_with_debug(self, frame: np.ndarray, sensitive: bool = False) -> Tuple[Optional[BowlDetection], np.ndarray]:
        """
        Detect bowl and return debug visualization.
        
        Args:
            frame: BGR image
            sensitive: If True, use more sensitive detection
        
        Returns:
            (detection, debug_frame)
        """
        detection = self.detect(frame, sensitive=sensitive)
        
        debug_frame = frame.copy()
        
        # Draw ROI rectangle
        cv2.rectangle(debug_frame, 
                     (self.roi_x1, self.roi_y1), 
                     (self.roi_x2, self.roi_y2), 
                     (255, 255, 0), 2)
        
        # Draw expected center
        cv2.circle(debug_frame, self.config.expected_center, 5, (0, 255, 255), -1)
        
        if detection:
            # Draw detected circle
            cv2.circle(debug_frame, 
                      (detection.center_x, detection.center_y), 
                      detection.radius, 
                      (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(debug_frame, 
                      (detection.center_x, detection.center_y), 
                      3, (0, 0, 255), -1)
            
            # Add text
            text = f"Bowl: ({detection.center_x}, {detection.center_y}) r={detection.radius} conf={detection.confidence:.2f}"
            cv2.putText(debug_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(debug_frame, "No bowl detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return detection, debug_frame


def benchmark_detector(detector: BowlDetector, frames: list) -> dict:
    """Benchmark detector performance on a list of frames."""
    times = []
    detections = []
    
    for frame in frames:
        start = time.perf_counter()
        detection = detector.detect(frame)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        times.append(elapsed)
        detections.append(detection)
    
    return {
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'detection_rate': sum(1 for d in detections if d is not None) / len(detections),
        'detections': detections
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bowl Detector")
    parser.add_argument("--input", "-i", required=True, help="Input video or image path")
    parser.add_argument("--output", "-o", help="Output video or image path")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    parser.add_argument("--show", action="store_true", help="Display results")
    args = parser.parse_args()
    
    # Initialize detector
    detector = BowlDetector()
    
    # Check if input is image or video
    if args.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # Process single image
        frame = cv2.imread(args.input)
        if frame is None:
            print(f"Error: Could not read image {args.input}")
            exit(1)
        
        detection, debug_frame = detector.detect_with_debug(frame)
        
        if detection:
            print(f"Detected bowl at ({detection.center_x}, {detection.center_y}), "
                  f"radius={detection.radius}, confidence={detection.confidence:.3f}")
        else:
            print("No bowl detected")
        
        if args.output:
            cv2.imwrite(args.output, debug_frame)
            print(f"Saved result to {args.output}")
        
        if args.show:
            cv2.imshow("Bowl Detection", debug_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    else:
        # Process video
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open video {args.input}")
            exit(1)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        
        frames_for_benchmark = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if args.benchmark and frame_count < 100:
                frames_for_benchmark.append(frame.copy())
            
            detection, debug_frame = detector.detect_with_debug(frame)
            
            if writer:
                writer.write(debug_frame)
            
            if args.show:
                cv2.imshow("Bowl Detection", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        if args.benchmark and frames_for_benchmark:
            print("\n=== Benchmark Results ===")
            results = benchmark_detector(detector, frames_for_benchmark)
            print(f"Mean time: {results['mean_time_ms']:.2f} ms")
            print(f"Std time: {results['std_time_ms']:.2f} ms")
            print(f"Min time: {results['min_time_ms']:.2f} ms")
            print(f"Max time: {results['max_time_ms']:.2f} ms")
            print(f"Detection rate: {results['detection_rate']*100:.1f}%")
            print(f"FPS capability: {1000/results['mean_time_ms']:.0f} fps")
