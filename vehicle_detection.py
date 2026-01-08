"""
Vehicle Detection and Classification System
Using YOLOv8 for counting and identifying vehicle types at traffic intersections
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from datetime import datetime

class VehicleDetector:
    """
    Vehicle detection and counting system for traffic intersections
    """
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the vehicle detector
        
        Args:
            model_path: Path to YOLO model weights
        """
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Vehicle classes from COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Track vehicles to avoid double counting
        self.tracked_vehicles = {}
        self.next_vehicle_id = 0
        
        # Counting lines for each direction
        self.counting_lines = {}
        
        # Statistics for each intersection
        self.intersection_stats = defaultdict(lambda: {
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0,
            'total': 0
        })
        
        # Colors for different vehicle types
        self.colors = {
            'car': (0, 255, 0),        # Green
            'motorcycle': (255, 0, 0),  # Blue
            'bus': (0, 165, 255),       # Orange
            'truck': (0, 0, 255)        # Red
        }
    
    def setup_counting_line(self, intersection_id, line_position, direction='horizontal'):
        """
        Set up counting line for vehicle detection
        
        Args:
            intersection_id: ID of the intersection
            line_position: Position of the counting line (x or y coordinate)
            direction: 'horizontal' or 'vertical'
        """
        self.counting_lines[intersection_id] = {
            'position': line_position,
            'direction': direction
        }
    
    def detect_vehicles(self, frame, intersection_id):
        """
        Detect and classify vehicles in the frame
        
        Args:
            frame: Input video frame
            intersection_id: ID of the intersection being monitored
            
        Returns:
            Processed frame with annotations
        """
        # Run YOLO detection
        results = self.model(frame, conf=0.5, verbose=False)
        
        detected_vehicles = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Check if detected object is a vehicle
                if class_id in self.vehicle_classes:
                    vehicle_type = self.vehicle_classes[class_id]
                    
                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    detected_vehicles.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (center_x, center_y),
                        'type': vehicle_type,
                        'confidence': confidence
                    })
        
        return detected_vehicles
    
    def track_and_count(self, detected_vehicles, intersection_id):
        """
        Track vehicles and count them when they cross the counting line
        
        Args:
            detected_vehicles: List of detected vehicles
            intersection_id: ID of the intersection
        """
        if intersection_id not in self.counting_lines:
            return
        
        line_info = self.counting_lines[intersection_id]
        line_position = line_info['position']
        direction = line_info['direction']
        
        for vehicle in detected_vehicles:
            center_x, center_y = vehicle['center']
            vehicle_type = vehicle['type']
            
            # Check if vehicle crossed the counting line
            crossed = False
            if direction == 'horizontal':
                # Check if vehicle crossed horizontal line (y-axis)
                if abs(center_y - line_position) < 10:
                    crossed = True
            else:
                # Check if vehicle crossed vertical line (x-axis)
                if abs(center_x - line_position) < 10:
                    crossed = True
            
            if crossed:
                # Simple tracking: check if vehicle is new based on position
                vehicle_key = f"{intersection_id}_{center_x}_{center_y}_{vehicle_type}"
                
                # Check if we've already counted this vehicle recently
                current_time = time.time()
                is_new_vehicle = True
                
                for tracked_key, tracked_data in list(self.tracked_vehicles.items()):
                    # Remove old tracked vehicles (older than 2 seconds)
                    if current_time - tracked_data['timestamp'] > 2:
                        del self.tracked_vehicles[tracked_key]
                        continue
                    
                    # Check if this is the same vehicle (within 50 pixels)
                    tracked_x, tracked_y = tracked_data['position']
                    distance = np.sqrt((center_x - tracked_x)**2 + (center_y - tracked_y)**2)
                    
                    if distance < 50 and tracked_data['type'] == vehicle_type:
                        is_new_vehicle = False
                        break
                
                if is_new_vehicle:
                    # Count the vehicle
                    self.intersection_stats[intersection_id][vehicle_type] += 1
                    self.intersection_stats[intersection_id]['total'] += 1
                    
                    # Track the vehicle
                    self.tracked_vehicles[vehicle_key] = {
                        'position': (center_x, center_y),
                        'type': vehicle_type,
                        'timestamp': current_time
                    }
                    
                    print(f"[{intersection_id}] New {vehicle_type} detected! Total: {self.intersection_stats[intersection_id]['total']}")
    
    def draw_detections(self, frame, detected_vehicles, intersection_id):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input video frame
            detected_vehicles: List of detected vehicles
            intersection_id: ID of the intersection
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw counting line
        if intersection_id in self.counting_lines:
            line_info = self.counting_lines[intersection_id]
            line_position = line_info['position']
            direction = line_info['direction']
            
            if direction == 'horizontal':
                cv2.line(annotated_frame, (0, line_position), 
                        (frame.shape[1], line_position), (255, 0, 255), 2)
            else:
                cv2.line(annotated_frame, (line_position, 0), 
                        (line_position, frame.shape[0]), (255, 0, 255), 2)
        
        # Draw detections
        for vehicle in detected_vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_type = vehicle['type']
            confidence = vehicle['confidence']
            color = self.colors.get(vehicle_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{vehicle_type}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            center_x, center_y = vehicle['center']
            cv2.circle(annotated_frame, (center_x, center_y), 5, color, -1)
        
        return annotated_frame
    
    def draw_statistics(self, frame, intersection_id):
        """
        Draw vehicle count statistics on the frame
        
        Args:
            frame: Input video frame
            intersection_id: ID of the intersection
            
        Returns:
            Frame with statistics overlay
        """
        stats = self.intersection_stats[intersection_id]
        
        # Create statistics panel
        panel_height = 200
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        
        # Add title
        title = f"Intersection {intersection_id} - Vehicle Statistics"
        cv2.putText(panel, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add statistics
        y_offset = 60
        cv2.putText(panel, f"Total Vehicles: {stats['total']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(panel, f"Cars: {stats['car']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['car'], 2)
        
        y_offset += 30
        cv2.putText(panel, f"Motorcycles: {stats['motorcycle']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['motorcycle'], 2)
        
        y_offset += 30
        cv2.putText(panel, f"Buses: {stats['bus']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['bus'], 2)
        
        y_offset += 30
        cv2.putText(panel, f"Trucks: {stats['truck']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['truck'], 2)
        
        # Combine frame and panel
        combined = np.vstack([panel, frame])
        
        return combined
    
    def get_statistics(self, intersection_id):
        """
        Get current statistics for an intersection
        
        Args:
            intersection_id: ID of the intersection
            
        Returns:
            Dictionary with vehicle counts
        """
        return dict(self.intersection_stats[intersection_id])
    
    def reset_statistics(self, intersection_id=None):
        """
        Reset statistics for an intersection or all intersections
        
        Args:
            intersection_id: ID of the intersection (None for all)
        """
        if intersection_id:
            self.intersection_stats[intersection_id] = {
                'car': 0,
                'motorcycle': 0,
                'bus': 0,
                'truck': 0,
                'total': 0
            }
        else:
            self.intersection_stats.clear()
        
        self.tracked_vehicles.clear()


def main():
    """
    Main function to run vehicle detection on video streams
    """
    print("Initializing Vehicle Detection System...")
    
    # Initialize detector
    detector = VehicleDetector()
    
    # Setup counting lines for two intersections
    # Intersection 1: Counting line at y=400 (horizontal)
    detector.setup_counting_line('intersection_1', line_position=400, direction='horizontal')
    
    # Intersection 2: Counting line at y=350 (horizontal)
    detector.setup_counting_line('intersection_2', line_position=350, direction='horizontal')
    
    print("\nVehicle Detection System Ready!")
    print("=" * 60)
    print("This is a template. To use with real video:")
    print("1. Replace video_path_1 and video_path_2 with actual video files")
    print("2. Or replace with camera indices (0, 1, etc.) for live cameras")
    print("3. Adjust counting line positions based on your camera angles")
    print("=" * 60)
    
    # Example: Process video files (replace with your actual video paths)
    video_sources = {
        'intersection_1': 0,  # Replace with video file path or camera index
        'intersection_2': 1   # Replace with video file path or camera index
    }
    
    print("\nNote: This example uses camera indices 0 and 1")
    print("If you don't have two cameras, modify the code to use video files")
    print("Example: video_sources = {'intersection_1': 'path/to/video1.mp4'}")
    

if __name__ == "__main__":
    main()
