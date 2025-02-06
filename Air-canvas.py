import cv2
import numpy as np
import mediapipe as mp
import pytesseract
import pyttsx3


# Set the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Capture video
cap = cv2.VideoCapture(0)
width = int(cap.get(3))
height = int(cap.get(4))

# Create a blank canvas
canvas = np.zeros((height, width, 3), np.uint8)
previous_point = None
user_permission = False  # Drawing permission
selected_color = (255, 0, 0)  # Default color: Blue

# Define button positions
button_clear = (20, 1, 100, 50)  # (x, y, width, height)
button_draw = (140, 1, 100, 50)
button_output = (260, 1, 150, 50)

def check_button_click(x, y):
    global canvas, user_permission
    if button_clear[0] < x < button_clear[0] + button_clear[2] and button_clear[1] < y < button_clear[1] + button_clear[3]:
        canvas = np.zeros((height, width, 3), np.uint8)  # Clear canvas
    elif button_draw[0] < x < button_draw[0] + button_draw[2] and button_draw[1] < y < button_draw[1] + button_draw[3]:
        user_permission = not user_permission  # Toggle drawing
    elif button_output[0] < x < button_output[0] + button_output[2] and button_output[1] < y < button_output[1] + button_output[3]:
        generate_output()  # Generate output

def generate_output():
    global canvas
    extracted_text = pytesseract.image_to_string(canvas).strip()

    # Shape detection
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_detected = ""

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            shape_detected = "Triangle"
        elif len(approx) == 4:
            shape_detected = "Rectangle or Square"
        elif len(approx) > 4:
            shape_detected = "Circle"

    final_output = extracted_text if extracted_text else shape_detected

    if final_output:
        print("Recognized Output:", final_output)
        engine.say(final_output)
        engine.runAndWait()
    else:
        print("No recognizable text or shape detected.")

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cX, cY = int(index_finger_tip.x * width), int(index_finger_tip.y * height)
            cv2.circle(frame, (cX, cY), 10, (0, 0, 255), 2)

            # Check if finger is clicking a button
            check_button_click(cX, cY)

            if user_permission and previous_point:
                cv2.line(canvas, previous_point, (cX, cY), selected_color, 3)
            previous_point = (cX, cY)
    else:
        previous_point = None

    # Draw UI Buttons
    cv2.rectangle(frame, (button_clear[0], button_clear[1]), 
                  (button_clear[0] + button_clear[2], button_clear[1] + button_clear[3]), (0, 0, 255), -1)
    cv2.putText(frame, "CLEAR", (button_clear[0] + 20, button_clear[1] + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(frame, (button_draw[0], button_draw[1]), 
                  (button_draw[0] + button_draw[2], button_draw[1] + button_draw[3]), (0, 255, 0), -1)
    cv2.putText(frame, "DRAW", (button_draw[0] + 25, button_draw[1] + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(frame, (button_output[0], button_output[1]), 
                  (button_output[0] + button_output[2], button_output[1] + button_output[3]), (255, 0, 0), -1)
    cv2.putText(frame, "OUTPUT", (button_output[0] + 30, button_output[1] + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Combine frame and canvas
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, canvas_binary = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
    canvas_binary = cv2.cvtColor(canvas_binary, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, canvas_binary)
    frame = cv2.bitwise_or(frame, canvas)

    # Display frame
    cv2.imshow("Air Canvas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
