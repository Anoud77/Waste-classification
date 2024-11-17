#nano project.py
import lgpio as GPIO
import time
import cv2
import numpy as np
import onnxruntime as ort
from gpiozero import LED
import requests  # Import the requests library for HTTP requests

# Define your class names and corresponding LEDs
class_names = ['glass', 'paper', 'plastic', 'waste']
leds = {
    "plastic": LED(6),
    "paper": LED(13),
    "glass": LED(19),
    "waste": LED(26)
}

# Set pins for each sensor
TRIG_PLASTIC = 27   # Associate pin 4 to TRIG for Plastic bin
ECHO_PLASTIC = 22  # Associate pin 17 to ECHO for Plastic bin

TRIG_GLASS = 4    # Associate pin 27 to TRIG for Glass bin
ECHO_GLASS = 17    # Associate pin 22 to ECHO for Glass bin

TRIG_PAPER = 23     # Associate pin 5 to TRIG for Paper bin
ECHO_PAPER = 24    # Associate pin 6 to ECHO for Paper bin

TRIG_WASTE = 14    # Associate pin 13 to TRIG for Waste bin
ECHO_WASTE = 15    # Associate pin 19 to ECHO for Waste bin

# Open the GPIO chip and set the GPIO direction
h = GPIO.gpiochip_open(0)
GPIO.gpio_claim_output(h, TRIG_PLASTIC)
GPIO.gpio_claim_input(h, ECHO_PLASTIC)

GPIO.gpio_claim_output(h, TRIG_GLASS)
GPIO.gpio_claim_input(h, ECHO_GLASS)

GPIO.gpio_claim_output(h, TRIG_PAPER)
GPIO.gpio_claim_input(h, ECHO_PAPER)

GPIO.gpio_claim_output(h, TRIG_WASTE)
GPIO.gpio_claim_input(h, ECHO_WASTE)

# Load the ONNX model
onnx_model_path = 'resNet50_Final.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# ThingSpeak API parameters
api_key = "B5KJ2VTY19NTV59U"  # Replace with your actual ThingSpeak Write API Key
url = f"https://api.thingspeak.com/update?api_key={api_key}"

# Function to send data to ThingSpeak
def send_to_thingspeak(dist_plastic, dist_glass, dist_paper, dist_waste):
    # Define the payload with your field values for each distance
    payload = {
        'field1': dist_plastic,
        'field2': dist_glass,
        'field3': dist_paper,
        'field4': dist_waste
    }
    # Send the data to ThingSpeak
    try:
        response = requests.get(url, params=payload)
        if response.status_code == 200:
            print("Data sent to ThingSpeak successfully.")
        else:
            print(f"Failed to send data. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending data to ThingSpeak: {e}")

# Function to preprocess the input frame
def preprocess(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize to model input size (224x224 for ResNet)
    frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
    frame = np.transpose(frame, (2, 0, 1))  # Change to CHW format
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Function to get predictions
def predict(frame):
    input_data = preprocess(frame)  # Preprocess the frame
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}  # Prepare input for the model
    logits = ort_session.run(None, ort_inputs)[0]  # Perform inference
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Convert logits to probabilities
    predicted_class_idx = np.argmax(probabilities)  # Find the predicted class index
    highest_probability = probabilities[0][predicted_class_idx]  # Get the highest probability
    return predicted_class_idx, highest_probability  # Return predicted class index and probability

# Function to get distance from the ultrasonic sensor
def get_distance(TRIG, ECHO):
    GPIO.gpio_write(h, TRIG, 0)
    time.sleep(0.1)  # Reduced initial delay for quicker responses
    GPIO.gpio_write(h, TRIG, 1)
    time.sleep(0.00001)
    GPIO.gpio_write(h, TRIG, 0)
    
    pulse_start = 0
    pulse_end = 0
    timeout = 0.01  # Timeout to avoid infinite loop
    start_time = time.time()
    
    while GPIO.gpio_read(h, ECHO) == 0:
        pulse_start = time.time()
        if pulse_start - start_time > timeout:
            return -1  # Return -1 if there's no pulse
    
    start_time = time.time()
    while GPIO.gpio_read(h, ECHO) == 1:
        pulse_end = time.time()
        if pulse_end - start_time > timeout:
            return -1  # Return -1 if there's no pulse
    
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Speed of sound in cm/s
    distance = round(distance, 2)
    return distance

# Open webcam
cap = cv2.VideoCapture(0)

# Turn off all LEDs initially
for led in leds.values():
    led.off()

try:
    while True:
        # Get the distance for each sensor
        dist_plastic = get_distance(TRIG_PLASTIC, ECHO_PLASTIC)
        dist_glass = get_distance(TRIG_GLASS, ECHO_GLASS)
        dist_paper = get_distance(TRIG_PAPER, ECHO_PAPER)
        dist_waste = get_distance(TRIG_WASTE, ECHO_WASTE)
        
        # Print the distances for each sensor
        print(f"Plastic bin distance = {dist_plastic:.2f} cm")
        print(f"Glass bin distance = {dist_glass:.2f} cm")
        print(f"Paper bin distance = {dist_paper:.2f} cm")
        print(f"Waste bin distance = {dist_waste:.2f} cm")
        
        # Send data to ThingSpeak
        send_to_thingspeak(dist_plastic, dist_glass, dist_paper, dist_waste)
        
        # Capture frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if there is an issue with the webcam
        
        # Get predictions
        predicted_class_idx, highest_probability = predict(frame)
        
        # Display prediction based on confidence threshold
        if highest_probability >= 0.9:
            predicted_class = class_names[predicted_class_idx]
            label = f"Predicted: {predicted_class} ({highest_probability:.2f})"
            
            # Turn on the corresponding LED
            for cls, led in leds.items():
                if cls == predicted_class:
                    led.on()  # Turn on the LED for the predicted class
                else:
                    led.off()  # Turn off LEDs for other classes
        else:
            label = "Confidence too low"
            
            # Turn off all LEDs if confidence is low
            for led in leds.values():
                led.off()
        
        # Display the prediction on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Webcam', frame)  # Show the frame with the prediction
        
        time.sleep(1)  # Wait before the next iteration
    
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Measurement stopped by User")
finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.gpiochip_close(h)  # Clean up GPIO
    for led in leds.values():
        led.off()  # Ensure all LEDs are turned off when done

#python project.py