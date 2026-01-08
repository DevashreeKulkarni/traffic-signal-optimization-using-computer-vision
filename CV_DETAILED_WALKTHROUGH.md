# Computer Vision Vehicle Detection - Detailed Code Walkthrough

## OVERVIEW: The Complete Flow

```
Camera Feed → YOLOv8 Detection → Classification → Tracking → Counting → Algorithm
```

---

## STEP-BY-STEP PROCESS

### STEP 1: INITIALIZATION
**File:** `vehicle_detection.py` (Lines 18-60)

```python
def __init__(self, model_path='yolov8n.pt'):
    # Load pre-trained YOLO model
    self.model = YOLO(model_path)
    
    # Define which COCO classes are vehicles
    self.vehicle_classes = {
        2: 'car',        # COCO class 2 = car
        3: 'motorcycle', # COCO class 3 = motorcycle
        5: 'bus',        # COCO class 5 = bus
        7: 'truck'       # COCO class 7 = truck
    }
    
    # Storage for tracking and counting
    self.tracked_vehicles = {}      # Prevents double counting
    self.intersection_stats = {}    # Stores counts per intersection
```

**What happens:**
- YOLOv8 model loads (trained on 80 COCO classes)
- We filter to only track 4 vehicle types (classes 2, 3, 5, 7)
- Initialize empty tracking dictionaries

---

### STEP 2: FRAME CAPTURE
**File:** `run_detection.py` (Lines 37-55)

```python
def process_intersection(self, intersection_id, video_source, window_name):
    cap = cv2.VideoCapture(video_source)  # Open camera/video
    
    while self.running:
        ret, frame = cap.read()  # Read one frame (image)
        
        if not ret:
            break
        
        # Send frame to detection
        detected_vehicles = self.detector.detect_vehicles(frame, intersection_id)
```

**What happens:**
- OpenCV captures frames from camera (typically 30 fps = 30 frames/second)
- Each frame is a single image (e.g., 1920x1080 pixels)
- Frame is sent to YOLO for detection

---

### STEP 3: VEHICLE DETECTION (CV Part 1)
**File:** `vehicle_detection.py` (Lines 76-118)

```python
def detect_vehicles(self, frame, intersection_id):
    # YOLO processes the frame
    results = self.model(frame, conf=0.5, verbose=False)
    
    detected_vehicles = []
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Extract detection data
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
            confidence = box.conf[0].cpu().numpy()       # Confidence score
            class_id = int(box.cls[0].cpu().numpy())    # Object class ID
            
            # Filter: only process vehicles
            if class_id in self.vehicle_classes:
                vehicle_type = self.vehicle_classes[class_id]
                
                # Calculate vehicle center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                detected_vehicles.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (center_x, center_y),
                    'type': vehicle_type,
                    'confidence': confidence
                })
    
    return detected_vehicles
```

**What happens in detail:**

1. **YOLO Processing:**
   ```
   Input: Frame (image array)
   ↓
   YOLO Neural Network analyzes image
   ↓
   Output: Bounding boxes for all detected objects
   ```

2. **For each detection, YOLO returns:**
   - `x1, y1, x2, y2`: Bounding box corners
     ```
     (x1, y1) ← Top-left corner
     (x2, y2) ← Bottom-right corner
     ```
   - `confidence`: How sure YOLO is (0.0 to 1.0)
     - 0.5 = 50% confident
     - 0.95 = 95% confident
   - `class_id`: What object type (0-79 in COCO dataset)

3. **Filtering:**
   - Only keep detections where `class_id` is 2, 3, 5, or 7
   - This filters out people, bicycles, animals, etc.

4. **Center Point Calculation:**
   ```python
   center_x = (x1 + x2) / 2
   center_y = (y1 + y2) / 2
   ```
   Used for tracking movement

---

### STEP 4: VEHICLE CLASSIFICATION (CV Part 2)
**How type identification works:**

```python
# YOLO already classified the object!
class_id = int(box.cls[0].cpu().numpy())

# We just map COCO class numbers to our names
if class_id == 2: vehicle_type = 'car'
if class_id == 3: vehicle_type = 'motorcycle'
if class_id == 5: vehicle_type = 'bus'
if class_id == 7: vehicle_type = 'truck'
```

**Important:** YOLO does the classification automatically!
- It was pre-trained on thousands of images
- It learned patterns:
  - Cars: 4 wheels, smaller size, common shape
  - Motorcycles: 2 wheels, thin profile
  - Buses: Large, rectangular, tall
  - Trucks: Large, cargo area, varies in shape

We don't write classification logic - YOLO already knows!

---

### STEP 5: VEHICLE TRACKING (CV Part 3)
**File:** `vehicle_detection.py` (Lines 120-182)

```python
def track_and_count(self, detected_vehicles, intersection_id):
    line_position = self.counting_lines[intersection_id]['position']
    direction = self.counting_lines[intersection_id]['direction']
    
    for vehicle in detected_vehicles:
        center_x, center_y = vehicle['center']
        vehicle_type = vehicle['type']
        
        # CHECK: Did vehicle cross the counting line?
        crossed = False
        if direction == 'horizontal':
            if abs(center_y - line_position) < 10:  # Within 10 pixels
                crossed = True
        
        if crossed:
            # CHECK: Have we counted this vehicle already?
            current_time = time.time()
            is_new_vehicle = True
            
            for tracked_key, tracked_data in self.tracked_vehicles.items():
                # Remove vehicles tracked >2 seconds ago
                if current_time - tracked_data['timestamp'] > 2:
                    del self.tracked_vehicles[tracked_key]
                    continue
                
                # Calculate distance from previously tracked position
                tracked_x, tracked_y = tracked_data['position']
                distance = sqrt((center_x - tracked_x)² + (center_y - tracked_y)²)
                
                # If same vehicle (within 50 pixels), don't count again
                if distance < 50 and tracked_data['type'] == vehicle_type:
                    is_new_vehicle = False
                    break
            
            if is_new_vehicle:
                # INCREMENT THE COUNT!
                self.intersection_stats[intersection_id][vehicle_type] += 1
                self.intersection_stats[intersection_id]['total'] += 1
                
                # Remember this vehicle to prevent double-counting
                self.tracked_vehicles[vehicle_key] = {
                    'position': (center_x, center_y),
                    'type': vehicle_type,
                    'timestamp': current_time
                }
```

**What happens:**

1. **Counting Line Check:**
   ```
   Frame:
   |------------------|
   |                  |
   |    🚗           |
   |================= | ← Counting line (y=400)
   |         🚌      |
   |                  |
   |------------------|
   
   If vehicle center touches line → potential count
   ```

2. **Anti-Double-Counting Logic:**
   ```
   Frame 1: Car at (500, 395) → Near line, count it
   Frame 2: Car at (505, 402) → Still near line, but we saw it!
   
   Distance = sqrt((505-500)² + (402-395)²) = 8.6 pixels
   
   8.6 < 50 → Same car, DON'T count again
   ```

3. **Temporal Tracking:**
   ```
   Time 0:00 → Car crosses, count = 1, store position
   Time 0:01 → Car still visible, skip
   Time 0:02 → Car drives away
   Time 0:03 → Memory cleared (>2 seconds old)
   ```

---

### STEP 6: VISUAL FEEDBACK (CV Part 4)
**File:** `vehicle_detection.py` (Lines 184-245)

```python
def draw_detections(self, frame, detected_vehicles, intersection_id):
    annotated_frame = frame.copy()
    
    # Draw counting line
    cv2.line(annotated_frame, (0, line_position), 
             (frame.shape[1], line_position), (255, 0, 255), 2)
    
    for vehicle in detected_vehicles:
        x1, y1, x2, y2 = vehicle['bbox']
        vehicle_type = vehicle['type']
        confidence = vehicle['confidence']
        
        # Draw bounding box (rectangle around vehicle)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with type and confidence
        label = f"{vehicle_type}: {confidence:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw center point
        cv2.circle(annotated_frame, (center_x, center_y), 5, color, -1)
    
    return annotated_frame
```

**Visual output looks like:**
```
┌─────────────────────────────────────┐
│  Intersection 1                     │
│                                     │
│  ┌────────┐                        │
│  │  car   │ ← Bounding box         │
│  │ 0.87   │ ← Confidence           │
│  └────●───┘ ← Center point         │
│                                     │
│═══════════════════════════════════ │ ← Counting line
│                                     │
│      ┌──────────┐                  │
│      │   bus    │                  │
│      │  0.92    │                  │
│      └─────●────┘                  │
│                                     │
│  Stats: Cars=5, Buses=2, Total=7  │
└─────────────────────────────────────┘
```

---

### STEP 7: ALGORITHM INTEGRATION
**File:** `integration_example.py`

```python
def get_traffic_density(self, intersection_id):
    # Get counts from detector
    stats = self.detector.get_statistics(intersection_id)
    
    # Apply weights
    weighted_density = (
        stats['car'] * 1.0 +
        stats['motorcycle'] * 0.5 +
        stats['bus'] * 2.0 +
        stats['truck'] * 2.0
    )
    
    return {
        'total_vehicles': stats['total'],
        'weighted_density': weighted_density,
        'car_count': stats['car'],
        'motorcycle_count': stats['motorcycle'],
        'bus_count': stats['bus'],
        'truck_count': stats['truck']
    }
```

**Example calculation:**
```
Detected at Intersection 1:
- 5 cars
- 2 motorcycles  
- 1 bus
- 0 trucks

Weighted density = (5 × 1.0) + (2 × 0.5) + (1 × 2.0) + (0 × 2.0)
                 = 5 + 1 + 2 + 0
                 = 8.0

This goes to the algorithm for signal optimization!
```

---

## COMPLETE DATA FLOW EXAMPLE

```
TIME: 12:30:00
┌──────────────┐
│ Camera feeds │ → Frame captured (1920x1080 image)
└──────┬───────┘
       ↓
┌──────────────┐
│ YOLOv8 Model │ → Detects 3 objects in frame:
└──────┬───────┘   1. Bounding box (100,200,300,400), class=2, conf=0.85
       ↓           2. Bounding box (500,150,700,350), class=5, conf=0.92
                   3. Bounding box (800,300,900,500), class=2, conf=0.78
┌──────────────┐
│ Classification│ → class 2 = car, class 5 = bus
└──────┬───────┘   Result: 2 cars, 1 bus
       ↓
┌──────────────┐
│   Tracking   │ → Check positions against counting line
└──────┬───────┘   Car 1 center: (200, 398) ← Crossed! Count it
       ↓           Bus center: (600, 250) ← Not at line yet
                   Car 2 center: (850, 410) ← Already counted (tracked)
┌──────────────┐
│   Counting   │ → New count: +1 car
└──────┬───────┘   Total: Cars=6, Motorcycles=0, Buses=1, Trucks=0
       ↓
┌──────────────┐
│  Algorithm   │ → Weighted density = (6×1.0)+(0×0.5)+(1×2.0)+(0×2.0) = 8.0
└──────┬───────┘   Priority score = 8.0×2 + 8 + 5 = 29
       ↓           Decision: Green time = 27 seconds
┌──────────────┐
│ Traffic Light│ → Set intersection to GREEN for 27 seconds
└──────────────┘
```

---

## KEY COMPUTER VISION CONCEPTS

### 1. **Object Detection (YOLO)**
- Neural network trained on millions of images
- Predicts bounding boxes + class + confidence in one pass
- Very fast: can process 30+ frames per second

### 2. **Bounding Box**
```
(x1, y1) = top-left corner
(x2, y2) = bottom-right corner

     (x1,y1)
        ┌─────────┐
        │ Vehicle │
        │         │
        └─────────┘
              (x2,y2)
```

### 3. **Center Point Tracking**
```
Used instead of full object tracking because:
- Simpler (just x,y coordinates)
- Faster (no complex matching)
- Sufficient for counting (just need to know if same vehicle)
```

### 4. **Temporal Threshold (2 seconds)**
```
Prevents counting the same vehicle multiple times as it passes through
the detection zone over several frames.
```

### 5. **Distance Threshold (50 pixels)**
```
If new detection is <50 pixels from previous → same vehicle
If >50 pixels → different vehicle (or same vehicle moved far)
```

---

## WHY THIS APPROACH WORKS

1. **YOLOv8 is pre-trained** - We don't need to train it ourselves
2. **COCO dataset includes vehicles** - Already knows car vs bus vs truck
3. **Simple counting logic** - Just check if center crosses line
4. **Anti-double-counting** - Temporal + spatial tracking prevents errors
5. **Weighted system** - Algorithm gets meaningful density values

---

## SUMMARY

**Detection:** YOLOv8 finds vehicles in image
**Classification:** YOLO tells us the type (car/bus/truck/motorcycle)
**Tracking:** We remember positions to avoid double-counting
**Counting:** When center crosses line AND it's new → count++
**Algorithm:** Uses counts to optimize traffic signals

The CV does the hard work (detection/classification), we just handle counting logic!
