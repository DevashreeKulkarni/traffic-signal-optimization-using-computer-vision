"""
Run Vehicle Detection System for Two Intersections
Processes video streams and displays real-time vehicle counting and classification
"""

import cv2
import numpy as np
from vehicle_detection import VehicleDetector
import threading
import time
from datetime import datetime


class MultiIntersectionManager:
    """
    Manages vehicle detection for multiple intersections simultaneously
    """
    
    def __init__(self):
        self.detector = VehicleDetector()
        self.running = False
        self.capture_threads = []
        
    def process_intersection(self, intersection_id, video_source, window_name):
        """
        Process video stream for a single intersection
        
        Args:
            intersection_id: ID of the intersection
            video_source: Video file path or camera index
            window_name: Name for the display window
        """
        # Open video source
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source for {intersection_id}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{intersection_id.upper()} Video Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                print(f"{intersection_id}: End of video stream")
                break
            
            frame_count += 1
            
            # Detect vehicles
            detected_vehicles = self.detector.detect_vehicles(frame, intersection_id)
            
            # Track and count vehicles
            self.detector.track_and_count(detected_vehicles, intersection_id)
            
            # Draw detections
            annotated_frame = self.detector.draw_detections(frame, detected_vehicles, intersection_id)
            
            # Add statistics overlay
            display_frame = self.detector.draw_statistics(annotated_frame, intersection_id)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, f"Time: {timestamp}", (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow(window_name, display_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('r'):
                # Reset statistics for this intersection
                self.detector.reset_statistics(intersection_id)
                print(f"\n{intersection_id}: Statistics reset")
        
        # Release resources
        cap.release()
        cv2.destroyWindow(window_name)
        print(f"\n{intersection_id}: Stream closed")
    
    def run(self, video_sources):
        """
        Run detection for multiple intersections
        
        Args:
            video_sources: Dictionary mapping intersection_id to video source
        """
        print("\n" + "="*70)
        print("STARTING MULTI-INTERSECTION VEHICLE DETECTION SYSTEM")
        print("="*70)
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to reset statistics")
        print("\n" + "="*70 + "\n")
        
        self.running = True
        threads = []
        
        # Create thread for each intersection
        for intersection_id, video_source in video_sources.items():
            window_name = f"Intersection {intersection_id.split('_')[1]}"
            thread = threading.Thread(
                target=self.process_intersection,
                args=(intersection_id, video_source, window_name)
            )
            thread.daemon = True
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Print final statistics
        self.print_final_statistics(video_sources.keys())
        
        cv2.destroyAllWindows()
        print("\nSystem shutdown complete.")
    
    def print_final_statistics(self, intersection_ids):
        """
        Print final statistics for all intersections
        """
        print("\n" + "="*70)
        print("FINAL VEHICLE COUNT STATISTICS")
        print("="*70)
        
        total_all_intersections = 0
        
        for intersection_id in intersection_ids:
            stats = self.detector.get_statistics(intersection_id)
            print(f"\n{intersection_id.upper()}:")
            print(f"  Total Vehicles: {stats['total']}")
            print(f"  Cars:          {stats['car']}")
            print(f"  Motorcycles:   {stats['motorcycle']}")
            print(f"  Buses:         {stats['bus']}")
            print(f"  Trucks:        {stats['truck']}")
            
            total_all_intersections += stats['total']
        
        print(f"\nTOTAL VEHICLES (ALL INTERSECTIONS): {total_all_intersections}")
        print("="*70 + "\n")


def run_with_video_files():
    """
    Example: Run detection with video files
    """
    manager = MultiIntersectionManager()
    
    # Setup counting lines for intersections
    # Adjust these positions based on your video/camera angle
    manager.detector.setup_counting_line('intersection_1', line_position=400, direction='horizontal')
    manager.detector.setup_counting_line('intersection_2', line_position=350, direction='horizontal')
    
    # Video sources - replace with your actual video file paths
    video_sources = {
        'intersection_1': 'videos/intersection1.mp4',  # Replace with actual path
        'intersection_2': 'videos/intersection2.mp4'   # Replace with actual path
    }
    
    try:
        manager.run(video_sources)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        manager.running = False


def run_with_cameras():
    """
    Example: Run detection with live cameras
    """
    manager = MultiIntersectionManager()
    
    # Setup counting lines for intersections
    # Adjust these positions based on your camera angle
    manager.detector.setup_counting_line('intersection_1', line_position=400, direction='horizontal')
    manager.detector.setup_counting_line('intersection_2', line_position=350, direction='horizontal')
    
    # Camera sources - 0 and 1 are typical camera indices
    video_sources = {
        'intersection_1': 0,  # First camera
        'intersection_2': 1   # Second camera
    }
    
    try:
        manager.run(video_sources)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        manager.running = False


def run_with_sample_video():
    """
    Run detection with a single sample video on both intersections (for testing)
    """
    manager = MultiIntersectionManager()
    
    # Setup counting lines
    manager.detector.setup_counting_line('intersection_1', line_position=300, direction='horizontal')
    manager.detector.setup_counting_line('intersection_2', line_position=300, direction='horizontal')
    
    # Use the same video for both intersections (for demonstration)
    # Replace 'sample_traffic.mp4' with your actual video file
    video_sources = {
        'intersection_1': 'sample_traffic.mp4',
        'intersection_2': 'sample_traffic.mp4'
    }
    
    try:
        manager.run(video_sources)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        manager.running = False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VEHICLE DETECTION & CLASSIFICATION SYSTEM")
    print("For 2-Way Lane Roads with 2 Intersections")
    print("="*70)
    
    print("\nSelect mode:")
    print("1. Run with video files")
    print("2. Run with live cameras")
    print("3. Run with sample video (testing)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\nNote: Edit the video file paths in the script before running!")
        print("Opening video files mode...")
        run_with_video_files()
    elif choice == '2':
        print("\nNote: Make sure you have cameras connected!")
        print("Opening camera mode...")
        run_with_cameras()
    elif choice == '3':
        print("\nNote: Place a traffic video file named 'sample_traffic.mp4' in the same directory")
        print("Opening sample video mode...")
        run_with_sample_video()
    else:
        print("Invalid choice! Exiting...")
