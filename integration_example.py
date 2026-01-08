"""
Traffic Signal Optimization Algorithm
Integrates with vehicle detection system to optimize traffic signal timing.
"""

from vehicle_detection import VehicleDetector
import time
from datetime import datetime


class TrafficSignalOptimizer:
    """
    Optimizes traffic signals based on real-time vehicle detection data.
    Uses adaptive timing and priority scoring to reduce congestion.
    """
    
    def __init__(self):
        self.detector = VehicleDetector()
        self.setup_intersections()
        
        # Signal configuration
        self.signal_states = {
            'intersection_1': 'RED',
            'intersection_2': 'RED'
        }
        
        self.min_green_time = 10
        self.max_green_time = 60
        self.yellow_time = 3
        
    def setup_intersections(self):
        """Configure counting lines for each intersection"""
        self.detector.setup_counting_line('intersection_1', line_position=400, direction='horizontal')
        self.detector.setup_counting_line('intersection_2', line_position=350, direction='horizontal')
    
    def get_traffic_density(self, intersection_id):
        """
        Calculate traffic density for an intersection.
        Applies different weights to vehicle types based on road space and clearance time.
        """
        stats = self.detector.get_statistics(intersection_id)
        
        # Weight vehicles by their impact on traffic flow
        weighted_density = (
            stats['car'] * 1.0 +
            stats['motorcycle'] * 0.5 +  # Smaller footprint
            stats['bus'] * 2.0 +          # Larger, slower
            stats['truck'] * 2.0          # Larger, slower
        )
        
        return {
            'total_vehicles': stats['total'],
            'weighted_density': weighted_density,
            'car_count': stats['car'],
            'motorcycle_count': stats['motorcycle'],
            'bus_count': stats['bus'],
            'truck_count': stats['truck']
        }
    
    def calculate_green_time(self, intersection_id):
        """
        Calculate optimal green signal duration based on current traffic.
        Uses three-tier system: light, medium, and heavy traffic.
        """
        density = self.get_traffic_density(intersection_id)
        weighted_density = density['weighted_density']
        
        if density['total_vehicles'] == 0:
            return self.min_green_time
        
        # Base timing calculation
        time_per_vehicle = 2
        
        if weighted_density <= 5:
            # Light traffic
            green_time = self.min_green_time
        elif weighted_density <= 15:
            # Medium traffic
            green_time = self.min_green_time + (weighted_density * time_per_vehicle)
        else:
            # Heavy traffic (add 20% extra time)
            green_time = self.min_green_time + (weighted_density * time_per_vehicle * 1.2)
        
        # Extra clearance time for large vehicles
        if density['bus_count'] > 0:
            green_time += density['bus_count'] * 3
        
        if density['truck_count'] > 0:
            green_time += density['truck_count'] * 3
        
        return int(max(self.min_green_time, min(green_time, self.max_green_time)))
    
    def optimize_signals(self):
        """
        Main optimization routine.
        Determines which intersection should receive green signal based on priority scoring.
        """
        density_1 = self.get_traffic_density('intersection_1')
        density_2 = self.get_traffic_density('intersection_2')
        
        print("\n" + "="*70)
        print("TRAFFIC SIGNAL OPTIMIZATION")
        print("="*70)
        
        print(f"\nIntersection 1:")
        print(f"  Total: {density_1['total_vehicles']} vehicles")
        print(f"  Weighted Density: {density_1['weighted_density']:.1f}")
        print(f"  Breakdown: Cars={density_1['car_count']}, Motorcycles={density_1['motorcycle_count']}, "
              f"Buses={density_1['bus_count']}, Trucks={density_1['truck_count']}")
        
        print(f"\nIntersection 2:")
        print(f"  Total: {density_2['total_vehicles']} vehicles")
        print(f"  Weighted Density: {density_2['weighted_density']:.1f}")
        print(f"  Breakdown: Cars={density_2['car_count']}, Motorcycles={density_2['motorcycle_count']}, "
              f"Buses={density_2['bus_count']}, Trucks={density_2['truck_count']}")
        
        # Calculate priority scores
        score_1 = self._calculate_priority_score(density_1)
        score_2 = self._calculate_priority_score(density_2)
        
        print(f"\nPriority Scores:")
        print(f"  Intersection 1: {score_1:.2f}")
        print(f"  Intersection 2: {score_2:.2f}")
        
        # Select priority intersection
        if score_1 > score_2:
            priority_intersection = 'intersection_1'
            priority_name = "Intersection 1"
            priority_density = density_1
            non_priority_density = density_2
            score_diff = score_1 - score_2
        else:
            priority_intersection = 'intersection_2'
            priority_name = "Intersection 2"
            priority_density = density_2
            non_priority_density = density_1
            score_diff = score_2 - score_1
        
        green_time = self.calculate_green_time(priority_intersection)
        
        # Coordination: reduce cycle time if both intersections are busy
        if non_priority_density['weighted_density'] > 10 and score_diff < 5:
            green_time = min(green_time, 30)
            coordination = "Balanced (both busy)"
        else:
            coordination = "Standard"
        
        # Classify overall traffic level
        total_density = density_1['weighted_density'] + density_2['weighted_density']
        if total_density < 10:
            traffic_level = "LIGHT"
        elif total_density < 30:
            traffic_level = "MODERATE"
        else:
            traffic_level = "HEAVY"
        
        print(f"\nDecision:")
        print(f"  Priority: {priority_name}")
        print(f"  Green Time: {green_time} seconds")
        print(f"  Coordination: {coordination}")
        print(f"  Traffic Level: {traffic_level}")
        print("="*70)
        
        return {
            'priority_intersection': priority_intersection,
            'green_time': green_time,
            'density_1': density_1,
            'density_2': density_2,
            'score_1': score_1,
            'score_2': score_2,
            'traffic_level': traffic_level
        }
    
    def _calculate_priority_score(self, density):
        """
        Calculate priority score based on multiple factors:
        - Weighted density (base score)
        - Public transport presence (buses)
        - Large vehicle presence (trucks)
        - Congestion level (total vehicle count)
        """
        score = density['weighted_density'] * 2
        
        # Bus priority (public transport serves more people)
        if density['bus_count'] > 0:
            score += 8  # Base bus bonus
            score += density['bus_count'] * 5
        
        # Truck priority (need more time to clear)
        if density['truck_count'] > 0:
            score += density['truck_count'] * 3
        
        # Congestion prevention
        if density['total_vehicles'] > 15:
            score += 10
        elif density['total_vehicles'] > 10:
            score += 5
        
        return score
    
    def run_optimization_cycle(self, duration_seconds=60):
        """
        Run continuous optimization cycles for specified duration.
        In production, this would interface with actual traffic light controllers.
        """
        print("\nStarting Traffic Signal Optimization System")
        print(f"Running for {duration_seconds} seconds\n")
        
        start_time = time.time()
        cycle_count = 0
        
        while (time.time() - start_time) < duration_seconds:
            cycle_count += 1
            print(f"\n{'#'*70}")
            print(f"Cycle #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'#'*70}")
            
            decision = self.optimize_signals()
            
            # In production: send signal to traffic controller
            # set_traffic_light(decision['priority_intersection'], 'GREEN')
            # time.sleep(decision['green_time'])
            # set_traffic_light(decision['priority_intersection'], 'YELLOW')
            # time.sleep(self.yellow_time)
            # set_traffic_light(decision['priority_intersection'], 'RED')
            
            print(f"\nWaiting {decision['green_time']} seconds...")
            time.sleep(decision['green_time'])
            
            self.detector.reset_statistics()
        
        print(f"\n{'='*70}")
        print(f"Optimization cycle completed - {cycle_count} cycles executed")
        print(f"{'='*70}")


def main():
    """Example usage of the traffic signal optimizer"""
    
    print("Traffic Signal Optimization System")
    print("="*70)
    print("\nNote: This is a demonstration. Replace video sources with actual")
    print("camera feeds or video files for real deployment.\n")
    
    optimizer = TrafficSignalOptimizer()
    
    # Run for 60 seconds (adjust as needed)
    optimizer.run_optimization_cycle(duration_seconds=60)


if __name__ == "__main__":
    main()
