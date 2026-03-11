import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import re 

import os
MODEL_PATH = "model/scientific_model1.h5"

print(f"Loading {MODEL_PATH}...")
try:
    from tensorflow.keras.models import load_model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: '{MODEL_PATH}' not found. Did you run the training script?")
        exit()
    model = load_model(MODEL_PATH, compile=False)
    print("✅ System Ready! (Zero-Lag + Batch Prediction + Algebra Fix + Pro UI)")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- NEW SMOOTH CONFIGURATION ---
WRITING_THICKNESS = 8  # Thick, smooth ink for the UI
SMOOTH_FONT = cv2.FONT_HERSHEY_DUPLEX

# The Lean 22-Class Mapping
CLASS_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '-', 12: '*', 13: '/',
    14: '=', 15: '.', 16: '(', 17: ')',
    18: '^', 19: 'v', 20: 'x', 21: 'y'
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)
dashboard_width, dashboard_height = 640, 480
dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
saved_box = np.zeros((dashboard_height, 350, 3), dtype=np.uint8) 

trajectory = []
recognized_expression = ""
saved_expressions = []
current_result = ""
writing_active = False
paused = False
previous_points = []
shake_threshold = 35 
shake_debounce_time = 0.5
last_shake_time = 0

last_prediction_time = 0
PREDICTION_INTERVAL = 0.5  

victory_triggered = False
fist_triggered = False
pinky_up_triggered = False

# --- IMPROVED MATH ENGINE (x/y FIX INCLUDED) ---
# --- IMPROVED MATH ENGINE (MULTIPLE SOLUTIONS FIX) ---
def solve_algebra(equation):
    try:
        equation = equation.replace('^', '**')
        # Handles numbers attached to variables or parentheses: 2x -> 2*x
        equation = re.sub(r'(\d)([xy])', r'\1*\2', equation)
        equation = re.sub(r'(\d)(\()', r'\1*\2', equation)
        
        # Check which variable was actually used in the equation
        var_used = 'y' if 'y' in equation else 'x'
        
        lhs, rhs = equation.split('=')
        
        solutions = [] # Create an empty list to store all correct answers
        
        for i in range(-100, 101):
            l_str = lhs.replace('x', f"({i})").replace('y', f"({i})")
            r_str = rhs.replace('x', f"({i})").replace('y', f"({i})")
            try:
                l_val = eval(l_str)
                r_val = eval(r_str)
            except: continue

            if abs(l_val - r_val) < 0.001:
                solutions.append(str(i)) # Add the answer to the list, DON'T stop!
        
        if solutions:
            # Join all found solutions together with a comma
            return f"{var_used} = {', '.join(solutions)}"
        else:
            return "No Int Solution"
            
    except Exception:
        return "Alg Error"

def evaluate_expression(expression):
    try:
        expr = expression.replace("X", "x").replace("Y", "y")
        expr = expr.replace("^", "**")             
        expr = re.sub(r'(\d)([xy])', r'\1*\2', expr)  
        expr = re.sub(r'(\d)(\()', r'\1*\2', expr)   
        
        if "=" in expr and ("x" in expr or "y" in expr):
            return solve_algebra(expr.replace("**", "^")) 
        
        if 'v' in expr: 
            expr = re.sub(r'v(\d+)', r'math.sqrt(\1)', expr)
            expr = expr.replace('v', 'math.sqrt')

        result = eval(expr, {"__builtins__": None}, {"math": math, "sqrt": math.sqrt, "abs": abs})
        
        if isinstance(result, float):
            return f"{result:.2f}"
        return str(result)
    except ZeroDivisionError: return "Infinity"
    except SyntaxError: return "Syntax Error"
    except Exception: return "Error"

# --- SHADOW MERGE ---
def aggressive_merge(contours):
    if not contours: return []
    boxes = [list(cv2.boundingRect(c)) for c in contours]
    boxes = [b for b in boxes if b[2] > 5 and b[3] > 5]
    boxes.sort(key=lambda b: b[0])
    
    merged = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if used[i]: continue
        curr = boxes[i]
        used[i] = True
        
        for j in range(i+1, len(boxes)):
            if used[j]: continue
            compare = boxes[j]
            if compare[0] > (curr[0] + curr[2]) + 5: break 

            x_start = max(curr[0], compare[0])
            x_end = min(curr[0]+curr[2], compare[0]+compare[2])
            overlap = x_end - x_start
            
            if overlap > 0 and abs(curr[1] - compare[1]) < 60:
                nx = min(curr[0], compare[0])
                ny = min(curr[1], compare[1])
                nw = max(curr[0]+curr[2], compare[0]+compare[2]) - nx
                nh = max(curr[1]+curr[3], compare[1]+compare[3]) - ny
                curr = [nx, ny, nw, nh]
                used[j] = True
        merged.append(curr)
    return merged

# --- MASSIVELY OPTIMIZED BATCH PREDICTION ---
def predict_chars_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return ""

    final_boxes = aggressive_merge(contours)
    if not final_boxes: return ""
    
    rois = []
    valid_boxes = []
    
    # Gather all pieces of the equation first
    for x, y, w, h in final_boxes:
        pad = 10
        roi = gray[max(0, y-pad):min(gray.shape[0], y+h+pad), 
                   max(0, x-pad):min(gray.shape[1], x+w+pad)]
        if roi.size == 0: continue

        resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        processed = (resized / 255.0).reshape(28, 28, 1)
        rois.append(processed)
        valid_boxes.append((w, h))

    if not rois: return ""

    # Feed ALL images to the AI in one massive batch (Fixes the lag)
    batch = np.array(rois)
    predictions = model.predict(batch, verbose=0)
    
    detected_string = ""
    for i, pred in enumerate(predictions):
        answer_index = np.argmax(pred)
        char = CLASS_MAPPING.get(answer_index, '?')
        w, h = valid_boxes[i]
        
        if char == '-' and (w / h < 2.5): 
            char = '='
            
        detected_string += char
        
    return detected_string

# --- GESTURES ---
def detect_shake(points):
    if len(points) < 25: return False
    displacements = [math.hypot(p1[0] - p2[0], p1[1] - p2[1]) for p1, p2 in zip(points, points[1:])]
    return (sum(displacements) / len(displacements)) > shake_threshold if displacements else False

def is_victory(lm):
    return (lm.landmark[8].y < lm.landmark[6].y and lm.landmark[12].y < lm.landmark[10].y and
            lm.landmark[16].y > lm.landmark[14].y and lm.landmark[20].y > lm.landmark[18].y)

def is_fist(lm):
    return all(lm.landmark[tip].y > lm.landmark[pip].y for tip, pip in [(8,6), (12,10), (16,14), (20,18)])

def is_pinky_up(lm):
    return (lm.landmark[20].y < lm.landmark[18].y and lm.landmark[8].y > lm.landmark[6].y and
            lm.landmark[12].y > lm.landmark[10].y and lm.landmark[16].y > lm.landmark[14].y)

def clear_writing():
    global trajectory, recognized_expression
    trajectory.clear()
    recognized_expression = ""

def delete_last_saved_expression():
    global saved_expressions
    if saved_expressions: saved_expressions.pop()

def clear_everything():
    global saved_expressions, current_result
    clear_writing()
    saved_expressions.clear()
    current_result = ""

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    dashboard[:] = (18, 18, 22) 

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            ix = int(hand_landmarks.landmark[8].x * frame.shape[1])
            iy = int(hand_landmarks.landmark[8].y * frame.shape[0])
            tx = int(hand_landmarks.landmark[4].x * frame.shape[1])
            ty = int(hand_landmarks.landmark[4].y * frame.shape[0])

            # SHAKE
            previous_points.append((ix, iy))
            if len(previous_points) > 30: previous_points.pop(0)
            if current_time - last_shake_time > shake_debounce_time and detect_shake(previous_points):
                clear_writing()
                previous_points.clear()
                last_shake_time = current_time
                continue

            # VICTORY
            if is_victory(hand_landmarks):
                if not victory_triggered and recognized_expression:
                    saved_expressions.append(recognized_expression)
                    clear_writing() 
                    victory_triggered = True
            else: victory_triggered = False

            # FIST
            if is_fist(hand_landmarks):
                if not fist_triggered and saved_expressions:
                    current_result = evaluate_expression("".join(saved_expressions))
                    fist_triggered = True
                    clear_writing()
            else: fist_triggered = False

            # PINKY
            if is_pinky_up(hand_landmarks):
                if not pinky_up_triggered:
                    delete_last_saved_expression()
                    pinky_up_triggered = True
                continue
            else: pinky_up_triggered = False

            # WRITING (Pinch)
            dist = math.hypot(ix - tx, iy - ty)
            if dist < 40:
                if paused: trajectory.append([])
                paused = False
                writing_active = True
                cv2.circle(frame, (ix, iy), 6, (0, 255, 255), -1) 
                
                if trajectory and (not trajectory[-1] or trajectory[-1][-1] != (ix, iy)):
                    trajectory[-1].append((ix, iy))
                elif not trajectory:
                    trajectory.append([(ix, iy)])
            else:
                paused = True
                writing_active = False

    # --- ZERO-LAG POLYLINE DRAWING (UI Canvas) ---
    for segment in trajectory:
        if len(segment) > 1:
            pts = np.array(segment, np.int32).reshape((-1, 1, 2))
            # Smooth, beautiful lines for the human eye
            cv2.polylines(frame, [pts], False, (0, 255, 120), WRITING_THICKNESS, cv2.LINE_AA)
            cv2.polylines(dashboard, [pts], False, (255, 255, 255), WRITING_THICKNESS, cv2.LINE_AA)

    # --- SMART IDLE PREDICTION (Hidden AI Canvas) ---
    if not writing_active and len(trajectory) > 0 and any(len(s) > 1 for s in trajectory):
        if current_time - last_prediction_time > PREDICTION_INTERVAL:
            all_points = [p for seg in trajectory for p in seg]
            if all_points:
                xs, ys = zip(*all_points)
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                
                pad = 20
                xmin, ymin = max(0, xmin - pad), max(0, ymin - pad)
                xmax, ymax = min(dashboard_width, xmax + pad), min(dashboard_height, ymax + pad)
                
                if xmax > xmin and ymax > ymin:
                    # Purely black and white, zero anti-aliasing canvas just for the CNN
                    ocr_image = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.uint8)
                    
                    for segment in trajectory:
                        if len(segment) > 1:
                            shifted_segment = [(p[0] - xmin, p[1] - ymin) for p in segment]
                            pts = np.array(shifted_segment, np.int32).reshape((-1, 1, 2))
                            # CRITICAL: thickness=6, LINE_8 (Harsh edges to match training data)
                            cv2.polylines(ocr_image, [pts], False, (255, 255, 255), 6, cv2.LINE_8)
                    
                    recognized_expression = predict_chars_from_image(ocr_image)
            last_prediction_time = current_time

    # --- PRO UI RENDERING ---
    saved_box[:] = (12, 12, 14)
    cv2.rectangle(saved_box, (0, 0), (saved_box.shape[1], 60), (30, 30, 35), -1)
    cv2.putText(saved_box, "MEMORY BANK", (20, 40), SMOOTH_FONT, 0.8, (255, 180, 0), 1, cv2.LINE_AA)
    
    y_offset = 100
    for i, expr in enumerate(saved_expressions):
        display_expr = expr if len(expr) <= 25 else expr[:22] + "..."
        cv2.putText(saved_box, f"> {display_expr}", (20, y_offset), SMOOTH_FONT, 0.7, (200, 255, 200), 1, cv2.LINE_AA)
        y_offset += 35

    cv2.rectangle(dashboard, (0, 0), (dashboard_width, 70), (30, 30, 35), -1)
    cv2.putText(dashboard, "DASHBOARD", (20, 45), SMOOTH_FONT, 0.9, (0, 255, 255), 1, cv2.LINE_AA)
    
    footer_y = dashboard_height - 140
    cv2.rectangle(dashboard, (0, footer_y), (dashboard_width, dashboard_height), (25, 25, 30), -1)
    cv2.putText(dashboard, f"DETECTED: {recognized_expression}", (20, footer_y + 50), SMOOTH_FONT, 0.9, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(dashboard, f"RESULT:   {current_result}", (20, footer_y + 100), SMOOTH_FONT, 1.1, (0, 200, 255), 2, cv2.LINE_AA)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame) 
    
    cv2.putText(frame, "Pinch: Write | Victory: Save", (20, 40), SMOOTH_FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Fist: Evaluate | Pinky: Undo", (20, 75), SMOOTH_FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Shake Hand: Erase Current Ink", (20, 110), SMOOTH_FONT, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Air Writing - Live Camera", frame)
    cv2.imshow("Air Writing - AI Vision", dashboard)
    cv2.imshow("Air Writing - Memory", saved_box)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'): current_result = evaluate_expression("".join(saved_expressions))
    elif key == ord('c'): clear_everything()
    elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()