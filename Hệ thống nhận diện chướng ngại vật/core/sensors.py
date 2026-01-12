# core/sensors.py - Quản lý cảm biến và dữ liệu phương tiện
import random
import time
import threading
import math
from datetime import datetime

class GPSSensor:
    """Mô phỏng cảm biến GPS"""
    def __init__(self, simulated=True, initial_location=None):
        self.simulated = simulated
        self.running = False
        self.update_thread = None
        
        # Dữ liệu GPS
        self.location = initial_location or (10.8231, 106.6297)  # Default: HCM City
        self.speed = 0.0  # km/h
        self.heading = 0.0  # degrees
        self.altitude = 0.0  # meters
        self.satellites = 12
        self.hdop = 1.2  # Horizontal dilution of precision
        
        # Lịch sử
        self.history = []
        self.max_history = 1000
        
        # Simulation parameters
        self.simulation_speed = 60  # km/h
        self.simulation_heading = 0
        self.route_points = []
        self.current_route_index = 0
        
        # Accuracy
        self.accuracy = 5.0  # meters
        
        # Thời gian
        self.last_update = datetime.now()
        
        # Khởi tạo route mẫu nếu simulated
        if simulated:
            self._init_sample_route()
    
    def _init_sample_route(self):
        """Khởi tạo route mẫu để simulation"""
        # Route vòng tròn đơn giản
        center_lat, center_lon = self.location
        
        # Tạo các điểm trên vòng tròn
        num_points = 36  # Mỗi 10 độ
        radius = 0.01  # ~1km
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            lat = center_lat + radius * math.cos(angle)
            lon = center_lon + radius * math.sin(angle)
            self.route_points.append((lat, lon))
    
    def start(self):
        """Bắt đầu cập nhật dữ liệu GPS"""
        if self.running:
            return True
        
        self.running = True
        
        if self.simulated:
            self.update_thread = threading.Thread(target=self._simulation_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            print("📍 GPS Simulation started")
        else:
            # TODO: Kết nối với GPS thật
            print("📍 GPS Real sensor started")
        
        return True
    
    def stop(self):
        """Dừng cập nhật dữ liệu GPS"""
        self.running = False
        
        if self.update_thread is not None:
            self.update_thread.join(timeout=2.0)
        
        print("📍 GPS stopped")
    
    def _simulation_loop(self):
        """Vòng lặp simulation GPS"""
        while self.running:
            try:
                self._update_simulation()
                time.sleep(1.0)  # Cập nhật mỗi giây
            except Exception as e:
                print(f"❌ Lỗi GPS simulation: {e}")
                time.sleep(5.0)
    
    def _update_simulation(self):
        """Cập nhật dữ liệu GPS mô phỏng"""
        # Di chuyển dọc theo route
        if self.route_points:
            target_lat, target_lon = self.route_points[self.current_route_index]
            
            # Tính khoảng cách và hướng
            current_lat, current_lon = self.location
            distance = self._calculate_distance(current_lat, current_lon, target_lat, target_lon)
            
            # Nếu đến gần điểm tiếp theo, chuyển điểm
            if distance < 0.001:  # ~100m
                self.current_route_index = (self.current_route_index + 1) % len(self.route_points)
                target_lat, target_lon = self.route_points[self.current_route_index]
            
            # Tính hướng
            self.heading = self._calculate_bearing(current_lat, current_lon, target_lat, target_lon)
            
            # Di chuyển về phía điểm đích
            speed_mps = self.simulation_speed * 1000 / 3600  # Chuyển km/h -> m/s
            distance_moved = speed_mps * 1.0  # Di chuyển trong 1 giây
            
            # Cập nhật vị trí
            self.location = self._move_towards(current_lat, current_lon, 
                                              target_lat, target_lon, 
                                              distance_moved)
            
            # Cập nhật tốc độ
            self.speed = self.simulation_speed + random.uniform(-5, 5)
        
        else:
            # Random walk nếu không có route
            self.location = (
                self.location[0] + random.uniform(-0.0001, 0.0001),
                self.location[1] + random.uniform(-0.0001, 0.0001)
            )
            self.heading = (self.heading + random.uniform(-10, 10)) % 360
            self.speed = random.uniform(40, 80)
        
        # Thêm nhiễu
        self.location = (
            self.location[0] + random.uniform(-0.00001, 0.00001),
            self.location[1] + random.uniform(-0.00001, 0.00001)
        )
        
        # Cập nhật thời gian
        self.last_update = datetime.now()
        
        # Lưu vào history
        self.history.append({
            'timestamp': self.last_update,
            'location': self.location,
            'speed': self.speed,
            'heading': self.heading
        })
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Tính khoảng cách giữa 2 điểm (đơn giản hóa)"""
        # Sử dụng công thức Haversine đơn giản hóa cho khoảng cách ngắn
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        return math.sqrt(dlat*dlat + dlon*dlon)
    
    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Tính góc phương vị giữa 2 điểm"""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def _move_towards(self, lat1, lon1, lat2, lon2, distance):
        """Di chuyển từ điểm 1 về phía điểm 2 một khoảng cách nhất định"""
        # Tính bearing
        bearing = math.radians(self._calculate_bearing(lat1, lon1, lat2, lon2))
        
        # Earth radius in meters
        R = 6371000
        
        # Chuyển đổi tọa độ
        lat1, lon1 = map(math.radians, [lat1, lon1])
        
        # Tính vị trí mới
        lat2 = math.asin(math.sin(lat1) * math.cos(distance/R) +
                        math.cos(lat1) * math.sin(distance/R) * math.cos(bearing))
        
        lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance/R) * math.cos(lat1),
                                math.cos(distance/R) - math.sin(lat1) * math.sin(lat2))
        
        return (math.degrees(lat2), math.degrees(lon2))
    
    def get_data(self):
        """Lấy dữ liệu GPS hiện tại"""
        return {
            'latitude': self.location[0],
            'longitude': self.location[1],
            'speed': self.speed,
            'heading': self.heading,
            'altitude': self.altitude,
            'satellites': self.satellites,
            'hdop': self.hdop,
            'accuracy': self.accuracy,
            'timestamp': self.last_update,
            'simulated': self.simulated
        }
    
    def set_simulation_speed(self, speed_kmh):
        """Đặt tốc độ simulation"""
        self.simulation_speed = max(0, speed_kmh)
        return self.simulation_speed
    
    def set_location(self, lat, lon):
        """Đặt vị trí thủ công (cho testing)"""
        self.location = (lat, lon)
        return self.location


class VehicleSensors:
    """Quản lý tất cả cảm biến phương tiện"""
    def __init__(self, config):
        self.config = config
        self.running = False
        
        # Các cảm biến
        self.gps = GPSSensor(
            simulated=config.USE_GPS_SIMULATION,
            initial_location=config.SIMULATED_LOCATION
        )
        
        # Dữ liệu phương tiện
        self.vehicle_data = {
            'speed': config.SIMULATED_SPEED,
            'rpm': 2000,
            'fuel_level': 80.0,  # %
            'engine_temp': 85.0,  # °C
            'brake_pressure': 10.0,  # bar
            'throttle_position': 25.0,  # %
            'gear': 'D',
            'odometer': 12345.6,  # km
            'tire_pressure': {
                'front_left': 2.4,
                'front_right': 2.4,
                'rear_left': 2.3,
                'rear_right': 2.3
            },
            'battery_voltage': 12.6,
            'lights': {
                'headlights': True,
                'brakelights': False,
                'turn_signal': 'off'  # off/left/right
            }
        }
        
        # CAN Bus simulation
        self.can_data = {}
        
        # Update thread
        self.update_thread = None
        self.update_interval = 0.1  # 10Hz
        
        # Callbacks
        self.data_callbacks = []
        
        # Khởi động GPS
        self.gps.start()
    
    def start(self):
        """Bắt đầu cập nhật dữ liệu cảm biến"""
        if self.running:
            return True
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        print("🚗 Vehicle sensors started")
        return True
    
    def stop(self):
        """Dừng cập nhật dữ liệu cảm biến"""
        self.running = False
        
        if self.update_thread is not None:
            self.update_thread.join(timeout=2.0)
        
        self.gps.stop()
        
        print("🚗 Vehicle sensors stopped")
    
    def _update_loop(self):
        """Vòng lặp cập nhật dữ liệu cảm biến"""
        while self.running:
            try:
                self._update_sensors()
                
                # Gọi callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(self.get_all_data())
                    except Exception as e:
                        print(f"❌ Lỗi sensor callback: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ Lỗi sensor update: {e}")
                time.sleep(1.0)
    
    def _update_sensors(self):
        """Cập nhật tất cả cảm biến"""
        # Lấy dữ liệu GPS
        gps_data = self.gps.get_data()
        
        # Cập nhật dữ liệu phương tiện từ GPS
        self.vehicle_data['speed'] = gps_data['speed']
        
        # Mô phỏng các thông số khác dựa trên tốc độ
        speed = gps_data['speed']
        
        # RPM dựa trên tốc độ và gear
        if speed < 10:
            self.vehicle_data['rpm'] = 800 + random.uniform(-50, 50)  # Idle
        else:
            self.vehicle_data['rpm'] = 1500 + speed * 20 + random.uniform(-100, 100)
            self.vehicle_data['rpm'] = min(self.vehicle_data['rpm'], 6000)
        
        # Giảm nhiên liệu theo thời gian
        fuel_consumption = speed * 0.01 * self.update_interval / 3600
        self.vehicle_data['fuel_level'] -= fuel_consumption
        self.vehicle_data['fuel_level'] = max(0, self.vehicle_data['fuel_level'])
        
        # Nhiệt độ động cơ
        base_temp = 80.0
        temp_increase = speed * 0.05 + self.vehicle_data['rpm'] * 0.001
        self.vehicle_data['engine_temp'] = base_temp + temp_increase + random.uniform(-2, 2)
        
        # Áp suất phanh (tăng khi phanh)
        if random.random() < 0.1:  # 10% chance of braking
            self.vehicle_data['brake_pressure'] = 30.0 + random.uniform(-5, 5)
        else:
            self.vehicle_data['brake_pressure'] = 10.0 + random.uniform(-2, 2)
        
        # Vị trí bướm ga
        self.vehicle_data['throttle_position'] = min(100, speed * 1.5 + random.uniform(-10, 10))
        
        # Tăng số km
        self.vehicle_data['odometer'] += speed * self.update_interval / 3600
        
        # Áp suất lốp (thay đổi theo nhiệt độ)
        temp_factor = self.vehicle_data['engine_temp'] / 100
        for tire in self.vehicle_data['tire_pressure']:
            self.vehicle_data['tire_pressure'][tire] = 2.4 + temp_factor * 0.1 + random.uniform(-0.05, 0.05)
        
        # Điện áp ắc quy
        if self.vehicle_data['rpm'] > 1000:
            self.vehicle_data['battery_voltage'] = 14.0 + random.uniform(-0.2, 0.2)
        else:
            self.vehicle_data['battery_voltage'] = 12.6 + random.uniform(-0.1, 0.1)
        
        # Đèn tín hiệu
        if random.random() < 0.05:  # 5% chance of turning
            self.vehicle_data['lights']['turn_signal'] = random.choice(['left', 'right'])
        elif random.random() < 0.05:  # 5% chance of stopping signal
            self.vehicle_data['lights']['turn_signal'] = 'off'
    
    def get_all_data(self):
        """Lấy tất cả dữ liệu cảm biến"""
        gps_data = self.gps.get_data()
        
        return {
            'gps': gps_data,
            'vehicle': self.vehicle_data.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_vehicle_data(self):
        """Lấy dữ liệu phương tiện"""
        return self.vehicle_data.copy()
    
    def get_gps_data(self):
        """Lấy dữ liệu GPS"""
        return self.gps.get_data()
    
    def register_callback(self, callback):
        """Đăng ký callback để nhận dữ liệu cảm biến"""
        if callback not in self.data_callbacks:
            self.data_callbacks.append(callback)
    
    def unregister_callback(self, callback):
        """Hủy đăng ký callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def set_vehicle_parameter(self, param, value):
        """Đặt thông số phương tiện thủ công"""
        if param in self.vehicle_data:
            if isinstance(self.vehicle_data[param], dict) and isinstance(value, dict):
                self.vehicle_data[param].update(value)
            else:
                self.vehicle_data[param] = value
            return True
        return False
    
    def trigger_event(self, event_type, data=None):
        """Kích hoạt sự kiện phương tiện"""
        events = {
            'hard_brake': {'brake_pressure': 50.0, 'description': 'Hard brake detected'},
            'sudden_acceleration': {'throttle_position': 80.0, 'description': 'Sudden acceleration'},
            'low_fuel': {'fuel_level': 10.0, 'description': 'Low fuel warning'},
            'high_temperature': {'engine_temp': 110.0, 'description': 'Engine overheating'},
            'tire_pressure_low': {'tire_pressure': {'front_left': 1.8}, 'description': 'Low tire pressure'},
        }
        
        if event_type in events:
            event_data = events[event_type]
            print(f"🚨 Vehicle event: {event_data['description']}")
            
            # Cập nhật dữ liệu
            for key, value in event_data.items():
                if key != 'description':
                    self.set_vehicle_parameter(key, value)
            
            return event_data
        
        return None