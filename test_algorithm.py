"""
Traffic Signal Optimization Algorithm - Test Suite
Tests the algorithm with simulated traffic data without requiring cameras or video files.
"""

import time
from datetime import datetime


class MockTrafficData:
    """Simulates traffic detection data for testing purposes"""
    
    def __init__(self):
        self.scenarios = {
            'light_traffic': {
                'intersection_1': {'cars': 3, 'motorcycles': 1, 'buses': 0, 'trucks': 0},
                'intersection_2': {'cars': 2, 'motorcycles': 0, 'buses': 0, 'trucks': 0}
            },
            'bus_waiting': {
                'intersection_1': {'cars': 5, 'motorcycles': 0, 'buses': 1, 'trucks': 0},
                'intersection_2': {'cars': 10, 'motorcycles': 0, 'buses': 0, 'trucks': 0}
            },
            'heavy_traffic': {
                'intersection_1': {'cars': 20, 'motorcycles': 0, 'buses': 2, 'trucks': 1},
                'intersection_2': {'cars': 8, 'motorcycles': 2, 'buses': 0, 'trucks': 0}
            },
            'both_busy': {
                'intersection_1': {'cars': 12, 'motorcycles': 0, 'buses': 1, 'trucks': 0},
                'intersection_2': {'cars': 15, 'motorcycles': 0, 'buses': 0, 'trucks': 0}
            },
            'rush_hour': {
                'intersection_1': {'cars': 25, 'motorcycles': 5, 'buses': 3, 'trucks': 2},
                'intersection_2': {'cars': 18, 'motorcycles': 3, 'buses': 1, 'trucks': 1}
            }
        }
        
        self.current_scenario = None
        self.current_data = {}
    
    def set_scenario(self, scenario_name):
        """Set the current traffic scenario"""
        if scenario_name in self.scenarios:
            self.current_scenario = scenario_name
            self.current_data = self.scenarios[scenario_name]
            return True
        return False
    
    def get_intersection_data(self, intersection_id):
        """Get traffic data for an intersection"""
        return self.current_data.get(intersection_id, {
            'cars': 0, 'motorcycles': 0, 'buses': 0, 'trucks': 0
        })


class TrafficSignalOptimizer:
    """Traffic signal optimization algorithm"""
    
    def __init__(self, mock_data=None):
        self.mock_data = mock_data or MockTrafficData()
        
        # Configuration parameters
        self.min_green_time = 10
        self.max_green_time = 60
        self.yellow_time = 3
        self.time_per_vehicle = 2
        self.bus_extra_time = 3
        self.truck_extra_time = 3
        
        # Vehicle weights for density calculation
        self.vehicle_weights = {
            'cars': 1.0,
            'motorcycles': 0.5,
            'buses': 2.0,
            'trucks': 2.0
        }
    
    def get_traffic_density(self, intersection_id):
        """Get traffic data for intersection"""
        return self.mock_data.get_intersection_data(intersection_id)
    
    def calculate_weighted_density(self, density):
        """Calculate weighted density based on vehicle types"""
        weighted = 0
        for vehicle_type, count in density.items():
            weight = self.vehicle_weights.get(vehicle_type, 1.0)
            weighted += count * weight
        return weighted
    
    def _calculate_priority_score(self, density):
        """Calculate priority score for an intersection"""
        score = 0
        
        # Calculate weighted density
        weighted_density = self.calculate_weighted_density(density)
        
        # Base score from weighted density
        score += weighted_density * 2
        
        # Public transport priority (buses serve many people)
        if density.get('buses', 0) > 0:
            score += 8  # Flat bonus for having any bus (public transport presence)
            score += density['buses'] * 5  # Additional per-bus multiplier
        
        # Large vehicle priority (trucks need special handling)
        if density.get('trucks', 0) > 0:
            score += density['trucks'] * 3
        
        # Total vehicles for congestion detection
        total_vehicles = sum(density.values())
        
        # Add urgency bonus for congested intersections
        if total_vehicles > 15:
            score += 10  # High urgency
        elif total_vehicles > 10:
            score += 5   # Medium urgency
        
        return score, weighted_density, total_vehicles
    
    def calculate_green_time(self, intersection_id):
        """Calculate optimal green time for an intersection"""
        density = self.get_traffic_density(intersection_id)
        
        # Calculate weighted density
        weighted_density = self.calculate_weighted_density(density)
        
        # Determine traffic level and calculate base time
        if weighted_density <= 5:
            # Light traffic
            traffic_level = "LIGHT"
            green_time = self.min_green_time
        elif weighted_density <= 15:
            # Medium traffic - proportional allocation
            traffic_level = "MODERATE"
            green_time = self.min_green_time + (weighted_density * self.time_per_vehicle)
        else:
            # Heavy traffic - extended time with boost
            traffic_level = "HEAVY"
            base_time = self.min_green_time + (weighted_density * self.time_per_vehicle)
            green_time = base_time * 1.2  # 20% boost for heavy traffic
        
        # Add extra time for buses (public transport priority)
        if density.get('buses', 0) > 0:
            green_time += density['buses'] * self.bus_extra_time
        
        # Add extra time for trucks (need more clearance time)
        if density.get('trucks', 0) > 0:
            green_time += density['trucks'] * self.truck_extra_time
        
        # Apply min/max constraints
        green_time = max(self.min_green_time, min(green_time, self.max_green_time))
        
        return green_time, traffic_level, weighted_density
    
    def optimize_signals(self):
        """Main optimization logic - decides which intersection gets priority"""
        
        # Get density data for both intersections
        density_1 = self.get_traffic_density('intersection_1')
        density_2 = self.get_traffic_density('intersection_2')
        
        # Calculate priority scores
        score_1, weighted_1, total_1 = self._calculate_priority_score(density_1)
        score_2, weighted_2, total_2 = self._calculate_priority_score(density_2)
        
        # Determine priority intersection
        if score_1 > score_2:
            priority_intersection = 'intersection_1'
            priority_density = density_1
            non_priority_density = weighted_2
        else:
            priority_intersection = 'intersection_2'
            priority_density = density_2
            non_priority_density = weighted_1
        
        # Calculate green time for priority intersection
        green_time, traffic_level, weighted_density = self.calculate_green_time(priority_intersection)
        
        # Coordination: If both intersections are busy, use shorter cycles
        score_difference = abs(score_1 - score_2)
        if non_priority_density > 10 and score_difference < 5:
            # Both intersections need attention - reduce cycle time for fairness
            green_time = min(green_time, 30)
            coordination = "BALANCED"
        else:
            coordination = "PRIORITY"
        
        return {
            'priority_intersection': priority_intersection,
            'green_time': green_time,
            'yellow_time': self.yellow_time,
            'traffic_level': traffic_level,
            'coordination': coordination,
            'scores': {
                'intersection_1': score_1,
                'intersection_2': score_2
            },
            'densities': {
                'intersection_1': weighted_1,
                'intersection_2': weighted_2
            },
            'vehicle_counts': {
                'intersection_1': density_1,
                'intersection_2': density_2
            }
        }
    
    def print_decision(self, decision):
        """Print the optimization decision in a readable format"""
        print("\n" + "="*70)
        print("TRAFFIC SIGNAL OPTIMIZATION DECISION")
        print("="*70)
        
        # Intersection 1 details
        density_1 = decision['vehicle_counts']['intersection_1']
        print(f"\nINTERSECTION 1:")
        print(f"   Vehicles: {sum(density_1.values())} total")
        print(f"   Breakdown: Cars={density_1['cars']}, Motorcycles={density_1['motorcycles']}, "
              f"Buses={density_1['buses']}, Trucks={density_1['trucks']}")
        print(f"   Weighted Density: {decision['densities']['intersection_1']:.1f}")
        print(f"   Priority Score: {decision['scores']['intersection_1']:.1f}")
        
        # Intersection 2 details
        density_2 = decision['vehicle_counts']['intersection_2']
        print(f"\nINTERSECTION 2:")
        print(f"   Vehicles: {sum(density_2.values())} total")
        print(f"   Breakdown: Cars={density_2['cars']}, Motorcycles={density_2['motorcycles']}, "
              f"Buses={density_2['buses']}, Trucks={density_2['trucks']}")
        print(f"   Weighted Density: {decision['densities']['intersection_2']:.1f}")
        print(f"   Priority Score: {decision['scores']['intersection_2']:.1f}")
        
        # Decision
        print(f"\nDECISION:")
        priority_label = "GREEN" if decision['priority_intersection'] == 'intersection_1' else "RED"
        other_label = "RED" if decision['priority_intersection'] == 'intersection_1' else "GREEN"
        
        print(f"   Priority: {decision['priority_intersection'].upper()} ({priority_label})")
        print(f"   Traffic Level: {decision['traffic_level']}")
        print(f"   Coordination Mode: {decision['coordination']}")
        print(f"   Green Time: {decision['green_time']:.1f} seconds")
        print(f"   Yellow Time: {decision['yellow_time']} seconds")
        
        print("\n" + "-"*70)


def run_test_scenarios():
    """Run all test scenarios"""
    
    print("\n" + "="*70)
    print("TRAFFIC SIGNAL OPTIMIZATION ALGORITHM - TEST SUITE")
    print("="*70)
    
    # Initialize
    mock_data = MockTrafficData()
    optimizer = TrafficSignalOptimizer(mock_data)
    
    # Test scenarios
    scenarios = [
        ('light_traffic', "Scenario 1: Light Traffic at Both Intersections"),
        ('bus_waiting', "Scenario 2: Bus Waiting (Public Transport Priority)"),
        ('heavy_traffic', "Scenario 3: Heavy Traffic with Buses and Trucks"),
        ('both_busy', "Scenario 4: Both Intersections Busy (Fairness Test)"),
        ('rush_hour', "Scenario 5: Rush Hour Peak Traffic")
    ]
    
    for scenario_name, description in scenarios:
        print(f"\n\n{'*'*70}")
        print(f"{description}")
        print('*'*70)
        
        # Set scenario
        mock_data.set_scenario(scenario_name)
        
        # Run optimization
        decision = optimizer.optimize_signals()
        
        # Print results
        optimizer.print_decision(decision)
        
        # Small delay for readability
        time.sleep(0.5)
    
    # Summary
    print("\n\n" + "="*70)
    print("ALGORITHM TEST COMPLETE")
    print("="*70)
    print("\nKey Features Tested:")
    print("  - Weighted density calculation")
    print("  - Priority scoring system")
    print("  - Bus/truck priority")
    print("  - Traffic level classification")
    print("  - Coordination between intersections")
    print("  - Min/max time constraints")
    print("\n" + "="*70)


if __name__ == "__main__":
    run_test_scenarios()
