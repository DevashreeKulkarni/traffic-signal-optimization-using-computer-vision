"""
Configuration file for Vehicle Detection System
Edit these parameters based on your camera setup and requirements
"""

# Model Configuration
MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',  # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
    'confidence_threshold': 0.5,   # Minimum confidence for detection (0.0 to 1.0)
    'device': 'cpu'                # 'cpu' or 'cuda' for GPU acceleration
}

# Intersection 1 Configuration
INTERSECTION_1_CONFIG = {
    'id': 'intersection_1',
    'name': 'Main Street & Oak Avenue',
    'video_source': 0,  # Camera index or video file path
    
    # Counting line configuration
    'counting_line': {
        'position': 400,           # Pixel position (y-coordinate for horizontal line)
        'direction': 'horizontal', # 'horizontal' or 'vertical'
        'offset': 10              # Tolerance in pixels
    },
    
    # Camera position
    'camera_height': 6.0,  # meters above ground
    'camera_angle': 30,     # degrees from horizontal
    
    # Display settings
    'window_name': 'Intersection 1 - Main & Oak',
    'display_position': (0, 0)  # (x, y) position on screen
}

# Intersection 2 Configuration
INTERSECTION_2_CONFIG = {
    'id': 'intersection_2',
    'name': 'Park Road & 5th Street',
    'video_source': 1,  # Camera index or video file path
    
    # Counting line configuration
    'counting_line': {
        'position': 350,
        'direction': 'horizontal',
        'offset': 10
    },
    
    # Camera position
    'camera_height': 5.5,  # meters above ground
    'camera_angle': 35,     # degrees from horizontal
    
    # Display settings
    'window_name': 'Intersection 2 - Park & 5th',
    'display_position': (960, 0)  # (x, y) position on screen
}

# Vehicle Classification Settings
VEHICLE_CLASSES = {
    2: {'name': 'car', 'weight': 1, 'priority': 1},
    3: {'name': 'motorcycle', 'weight': 0.5, 'priority': 2},
    5: {'name': 'bus', 'weight': 2, 'priority': 3},
    7: {'name': 'truck', 'weight': 2, 'priority': 3}
}

# Tracking Configuration
TRACKING_CONFIG = {
    'max_distance': 50,        # Maximum distance (pixels) for same vehicle
    'tracking_timeout': 2.0,   # Seconds before removing tracked vehicle
    'min_hits': 2,             # Minimum detections before counting
}

# Display Configuration
DISPLAY_CONFIG = {
    'show_bounding_boxes': True,
    'show_labels': True,
    'show_confidence': True,
    'show_center_points': True,
    'show_counting_line': True,
    'show_statistics': True,
    'statistics_panel_height': 200,
    'font_scale': 0.6,
    'font_thickness': 2
}

# Color Configuration (BGR format)
COLORS = {
    'car': (0, 255, 0),         # Green
    'motorcycle': (255, 0, 0),  # Blue
    'bus': (0, 165, 255),       # Orange
    'truck': (0, 0, 255),       # Red
    'counting_line': (255, 0, 255),  # Magenta
    'text_background': (0, 0, 0),     # Black
    'text_color': (255, 255, 255)     # White
}

# System Configuration
SYSTEM_CONFIG = {
    'save_video': False,              # Save processed video
    'output_directory': './output',   # Directory for saved videos
    'save_statistics': True,          # Save statistics to file
    'statistics_interval': 60,        # Save statistics every N seconds
    'log_detections': True,           # Log each detection
    'log_file': 'detection_log.txt'   # Log file path
}

# Alert Configuration
ALERT_CONFIG = {
    'enable_alerts': True,
    'traffic_density_threshold': 10,  # Alert when more than N vehicles detected
    'alert_interval': 300,            # Minimum seconds between alerts
    'emergency_vehicle_priority': True
}

# Export configuration as dictionary
CONFIG = {
    'model': MODEL_CONFIG,
    'intersection_1': INTERSECTION_1_CONFIG,
    'intersection_2': INTERSECTION_2_CONFIG,
    'vehicle_classes': VEHICLE_CLASSES,
    'tracking': TRACKING_CONFIG,
    'display': DISPLAY_CONFIG,
    'colors': COLORS,
    'system': SYSTEM_CONFIG,
    'alerts': ALERT_CONFIG
}
