from ultralytics import YOLO
import cv2
import serial
import serial.tools.list_ports
import time
from threading import Thread, Event, Lock
from queue import Queue
import numpy as np

# Define servo pins and their corresponding finger names and orientations
SERVO_CONFIG = {
    'pinky':  {'pin': 13, 'close': 180, 'open': 0, 'current': 90},
    'ring':   {'pin': 12, 'close': 180, 'open': 0, 'current': 90},
    'middle': {'pin': 11, 'close': 180, 'open': 0, 'current': 90},
    'index':  {'pin': 10, 'close': 180, 'open': 0, 'current': 90},
    'thumb':  {'pin': 9,  'close': 0,   'open': 180, 'current': 90}
}

class HandController:
    def __init__(self):
        self.arduino = None
        self.command_queue = Queue()
        self.movement_lock = Lock()
        self.stop_current = Event()
        self.command_thread = Thread(target=self._process_commands, daemon=True)
        self.command_thread.start()

    def _process_commands(self):
        """Background thread to process commands"""
        while True:
            commands = self.command_queue.get()
            if commands is None:
                break
            self._send_command(commands)
            time.sleep(0.05)  # Small delay between command sets

    def _send_command(self, commands):
        """Internal method to send commands to Arduino"""
        try:
            if self.arduino and self.arduino.is_open:
                command_str = ";".join(commands) + "\n"
                self.arduino.write(command_str.encode())
                self.arduino.flush()
        except serial.SerialException:
            self.connect()

    def connect(self):
        """Setup serial connection with Arduino"""
        arduino_port = self.find_arduino_port()
        if not arduino_port:
            return False
        
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.close()
            
            self.arduino = serial.Serial(arduino_port, 115200, timeout=1)
            time.sleep(0.5)  # Reduced wait time
            return True
        except serial.SerialException:
            return False

    def find_arduino_port(self):
        """Find the port where Arduino is connected"""
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if 'usbmodem' in port.device.lower() or 'usbserial' in port.device.lower():
                return port.device
        return None

    def make_gesture(self, gesture_type):
        """Queue gesture commands"""
        if not self.arduino or not self.arduino.is_open:
            return

        commands = []
        if gesture_type == "rock":
            commands = [f"{SERVO_CONFIG[finger]['pin']},{SERVO_CONFIG[finger]['close']}" 
                       for finger in SERVO_CONFIG]
        elif gesture_type == "paper":
            commands = [f"{SERVO_CONFIG[finger]['pin']},{SERVO_CONFIG[finger]['open']}" 
                       for finger in SERVO_CONFIG]
        elif gesture_type == "scissors":
            commands = [f"{SERVO_CONFIG[finger]['pin']},{SERVO_CONFIG[finger]['open'] if finger in ['index', 'middle'] else SERVO_CONFIG[finger]['close']}" 
                       for finger in SERVO_CONFIG]
        
        if commands:
            self.command_queue.put(commands)

    def close(self):
        """Clean up resources"""
        self.command_queue.put(None)  # Signal thread to stop
        if self.arduino:
            self.arduino.close()

class VideoProcessor:
    def __init__(self, model, hand_controller):
        self.model = model
        self.hand_controller = hand_controller
        self.last_detected_class = None
        self.last_detection_time = 0
        self.detection_cooldown = 1.0  # Seconds between gesture changes
        self.winning_move_mapping = {
            "paper": "scissors",
            "scissors": "rock",
            "rock": "paper"
        }

    def process_frame(self, frame):
        current_time = time.time()
        inference_frame = cv2.resize(frame, (640, 480))
        results = self.model(inference_frame, verbose=False)  # Disable verbose output
        result = results[0]
        annotated_frame = result.plot()
        
        if (hasattr(result, "boxes") and 
            result.boxes is not None and 
            len(result.boxes) > 0 and 
            current_time - self.last_detection_time > self.detection_cooldown):
            
            cls_tensor = result.boxes.cls[0]
            cls_id = int(cls_tensor.item())
            
            detected_class = None
            if hasattr(result, "names") and isinstance(result.names, dict):
                detected_class = result.names.get(cls_id, None)
            else:
                fallback = {0: "paper", 1: "rock", 2: "scissors"}
                detected_class = fallback.get(cls_id, None)
            
            if detected_class:
                detected_class = detected_class.lower()
                if detected_class != self.last_detected_class:
                    winning_move = self.winning_move_mapping.get(detected_class)
                    self.hand_controller.make_gesture(winning_move)
                    self.last_detected_class = detected_class
                    self.last_detection_time = current_time
                    
                    # Add text overlay
                    text = f"Play: {winning_move}"
                    cv2.putText(annotated_frame, text, 
                              (50, annotated_frame.shape[0] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        return annotated_frame

def main():
    # Initialize hand controller
    hand = HandController()
    if not hand.connect():
        print("Failed to connect to Arduino. Exiting.")
        return

    # Load the YOLO model
    model = YOLO('best.onnx')
    model.task = 'detect'
    model.conf = 0.05

    # Initialize video processor
    processor = VideoProcessor(model, hand)

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow('YOLO Inference - Rock Paper Scissors', cv2.WINDOW_NORMAL)
    cv2.moveWindow('YOLO Inference - Rock Paper Scissors', 50, 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = processor.process_frame(frame)
            cv2.imshow('YOLO Inference - Rock Paper Scissors', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand.close()

if __name__ == '__main__':
    main()
