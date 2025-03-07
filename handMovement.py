
import serial
import serial.tools.list_ports
import time

# Define servo pins and their corresponding finger names and orientations
SERVO_CONFIG = {
    'pinky':  {'pin': 13, 'close': 180, 'open': 0, 'current': 90},
    'ring':   {'pin': 12, 'close': 180, 'open': 0, 'current': 90},
    'middle': {'pin': 11, 'close': 180, 'open': 0, 'current': 90},
    'index':  {'pin': 10, 'close': 180, 'open': 0, 'current': 90},
    'thumb':  {'pin': 9,  'close': 0,   'open': 180, 'current': 90}  # Reversed orientation for thumb
}

def find_arduino_port():
    """Find the port where Arduino is connected"""
    ports = list(serial.tools.list_ports.comports())
    print("Available ports:")
    for port in ports:
        print(f"  - {port.device}")
    
    for port in ports:
        if 'usbmodem' in port.device.lower() or 'usbserial' in port.device.lower():
            return port.device
    return None

def setup_arduino():
    """Setup serial connection with Arduino"""
    arduino_port = find_arduino_port()
    if not arduino_port:
        print("Error: Arduino not found. Please check the connection.")
        return None
    
    try:
        print(f"Connecting to Arduino on port: {arduino_port}")
        arduino = serial.Serial(arduino_port, 9600, timeout=1)  # Back to 9600 baud
        time.sleep(2)  # Wait for Arduino to reset
        print("Connected successfully")
        return arduino
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
        return None

def move_servo_smooth(arduino, pin, start_angle, target_angle, steps=10):
    """Move a servo smoothly from start to target angle"""
    if start_angle == target_angle:
        return
    
    step_size = (target_angle - start_angle) / steps
    for step in range(steps + 1):
        current_angle = int(start_angle + (step_size * step))
        command = f"{pin},{current_angle}\n"
        arduino.write(command.encode())
        arduino.flush()
        time.sleep(0.03)  # Adjust this delay to control movement speed

def move_finger_smooth(arduino, finger, target_angle):
    """Move a specific finger's servo smoothly"""
    if finger in SERVO_CONFIG:
        pin = SERVO_CONFIG[finger]['pin']
        start_angle = SERVO_CONFIG[finger]['current']
        move_servo_smooth(arduino, pin, start_angle, target_angle)
        SERVO_CONFIG[finger]['current'] = target_angle

def make_rock(arduino):
    """Make rock gesture (closed fist) - rapid movement"""
    print("Making rock gesture...")
    # Close all fingers simultaneously
    for finger in SERVO_CONFIG:
        command = f"{SERVO_CONFIG[finger]['pin']},{SERVO_CONFIG[finger]['close']}\n"
        arduino.write(command.encode())
        arduino.flush()
    time.sleep(0.1)  # Small delay to allow servos to reach position

def make_paper(arduino):
    """Make paper gesture (open hand) - rapid movement"""
    print("Making paper gesture...")
    # Open all fingers simultaneously
    for finger in SERVO_CONFIG:
        command = f"{SERVO_CONFIG[finger]['pin']},{SERVO_CONFIG[finger]['open']}\n"
        arduino.write(command.encode())
        arduino.flush()
    time.sleep(0.1)  # Small delay to allow servos to reach position

def make_scissors(arduino):
    """Make scissors gesture - rapid movement"""
    print("Making scissors gesture...")
    # Set all fingers simultaneously
    for finger in SERVO_CONFIG:
        if finger in ['index', 'middle']:
            angle = SERVO_CONFIG[finger]['open']
        else:
            angle = SERVO_CONFIG[finger]['close']
        command = f"{SERVO_CONFIG[finger]['pin']},{angle}\n"
        arduino.write(command.encode())
        arduino.flush()
    time.sleep(0.1)  # Small delay to allow servos to reach position

def test_all_servos():
    """Test all servo movements"""
    arduino = setup_arduino()
    if not arduino:
        return
    
    try:
        while True:
            print("\n=== Rock Paper Scissors Menu ===")
            print("1. Rock")
            print("2. Paper")
            print("3. Scissors")
            print("4. Test Single Servo")
            print("5. Exit")
            
            choice = input("Choice (1-5): ")
            
            if choice == '1':
                make_rock(arduino)
            elif choice == '2':
                make_paper(arduino)
            elif choice == '3':
                make_scissors(arduino)
            elif choice == '4':
                pin = input("Enter servo pin (9-13): ")
                angle = input("Enter angle (0-180): ")
                try:
                    current_angle = 90  # Assume middle position
                    move_servo_smooth(arduino, int(pin), current_angle, int(angle))
                except ValueError:
                    print("Invalid input. Please enter numbers only.")
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        arduino.close()

if __name__ == "__main__":
    test_all_servos()
