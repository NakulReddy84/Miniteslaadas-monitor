import asyncio
import websockets
import json
import base64
import time
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import logging
import os
from functools import partial

# Configure basic logging to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- Configuration ---
MODEL_PATH = 'yolov8n.pt'  # YOLO model for detection
FRAME_WIDTH = 640  
FRAME_HEIGHT = 480 
WEBSOCKET_PORT = 8765

# Time-to-Collision (TTC) thresholds in seconds
USER_CONFIG = {
    'critical_ttc': 1.5,
    'warning_ttc': 3.0,
    'safe_ttc': 999.0,
    'detection_confidence': 0.25,
    'speed_calibration_factor': 0.5 
}

# Target classes: None means detect ALL 80 COCO classes
TARGET_CLASSES = None 

# --- Utility Functions and Logic ---

def update_user_config(new_critical, new_warning):
    """Updates the global USER_CONFIG based on client input."""
    global USER_CONFIG
    try:
        critical_val = float(new_critical)
        warning_val = float(new_warning)
        
        if 0.1 < critical_val < warning_val:
            USER_CONFIG['critical_ttc'] = critical_val
            USER_CONFIG['warning_ttc'] = warning_val
            logging.info(f"CONFIG UPDATED: Critical={critical_val}s, Warning={warning_val}s")
            return True
        else:
            logging.warning("Config update failed: Invalid threshold values.")
            return False
    except ValueError:
        logging.warning("Config update failed: Threshold values must be numbers.")
        return False

def base64_to_image(base64_data):
    """Converts a Base64 string to an OpenCV image (numpy array)."""
    try:
        # Robustness: Remove header if present
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]

        # Robustness: Add padding if missing
        missing_padding = len(base64_data) % 4
        if missing_padding:
            base64_data += '=' * (4 - missing_padding)

        img_bytes = base64.b64decode(base64_data)
        
        if len(img_bytes) == 0:
            return None

        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if img is None:
            logging.warning("DEBUG: Image decode returned None")
            return None
            
        if np.mean(img) < 5:
            logging.warning("DEBUG: Received completely black frame")
            
        return img
    except Exception as e:
        logging.error(f"Image decode error: {e}")
        return None

class RiskSmoother:
    """Smooths risk levels over a short window to prevent rapid flickering."""
    def __init__(self, history_len=5):
        self.history = []
        self.history_len = history_len
        self.risk_precedence = {'Safe': 0, 'Warning': 1, 'Critical': 2}
        self.risk_names = {0: 'Safe', 1: 'Warning', 2: 'Critical'}

    def update_and_get_risk(self, current_risk):
        current_precedence = self.risk_precedence.get(current_risk, 0)
        self.history.append(current_precedence)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        
        highest_precedence = max(self.history)
        return self.risk_names.get(highest_precedence, 'Safe')

class ObjectTracker:
    """Calculates TTC based on the apparent height change of the closest object."""
    
    def __init__(self):
        self.last_frame_detections = {} 
        self.min_dh_dt_threshold = 0.001 

    def _get_risk_level(self, ttc):
        """Determines the risk level based on the calculated TTC using active config."""
        epsilon = 0.001
        if ttc <= USER_CONFIG['critical_ttc'] + epsilon:
            return "Critical"
        elif ttc <= USER_CONFIG['warning_ttc'] + epsilon:
            return "Warning"
        else:
            return "Safe"

    def update_and_get_ttc(self, results, current_time, model_names):
        """Processes YOLO results, calculates TTC and Speed."""
        detections_output = []
        
        if not results or not results[0].boxes or results[0].boxes.xyxy.numel() == 0:
            self.last_frame_detections = {}
            return detections_output 

        boxes = results[0].boxes
        heights = boxes.xyxy.cpu().numpy()[:, 3] - boxes.xyxy.cpu().numpy()[:, 1]
        
        closest_box_index = np.argmax(heights)
        
        box = boxes.xyxy.cpu().numpy()[closest_box_index]
        cls = int(boxes.cls[closest_box_index].item())
        
        x1, y1, x2, y2 = map(int, box)
        current_height = y2 - y1
        
        ttc_result = {
            'ttc': USER_CONFIG['safe_ttc'], 
            'risk_level': 'Safe', 
            'bbox': [x1, y1, x2, y2], 
            'label': model_names.get(cls, "UNKNOWN"), 
            'id': 1,
            'speed_mph': 0.0
        }
        
        # TTC & Speed Calculation Logic
        if self.last_frame_detections:
            last_height = self.last_frame_detections['height']
            last_time = self.last_frame_detections['time']
            dt = current_time - last_time
            
            if dt > 0.001:
                dh_dt = (current_height - last_height) / dt
                
                if dh_dt > self.min_dh_dt_threshold:
                    ttc = current_height / dh_dt
                    
                    # Calculate estimated relative speed (Proxy)
                    speed_mph = dh_dt * USER_CONFIG['speed_calibration_factor']
                    
                    if ttc > 0 and ttc < USER_CONFIG['safe_ttc']:
                        ttc_result['ttc'] = ttc
                        ttc_result['risk_level'] = self._get_risk_level(ttc)
                        ttc_result['speed_mph'] = speed_mph

        self.last_frame_detections = {
            'time': current_time,
            'height': current_height
        }
        
        return [ttc_result]

# --- Core WebSocket Server Logic ---

async def websocket_server_logic(websocket, model, tracker, smoother):
    """The main handler logic for incoming WebSocket messages."""
    
    # Send initial configuration
    config = {
        'type': 'config',
        'thresholds': {'critical_ttc': USER_CONFIG['critical_ttc'], 'warning_ttc': USER_CONFIG['warning_ttc']},
        'model': MODEL_PATH,
        'frame_size': {'width': FRAME_WIDTH, 'height': FRAME_HEIGHT}
    }
    await websocket.send(json.dumps(config))
    logging.info("DEBUG 1: Config sent to client.")

    frame_count = 0

    try:
        async for message in websocket:
            current_time = time.time()
            data = json.loads(message)
            
            if data.get('type') == 'SETTINGS_UPDATE':
                if update_user_config(data['critical'], data['warning']):
                    config['thresholds'] = {'critical_ttc': USER_CONFIG['critical_ttc'], 'warning_ttc': USER_CONFIG['warning_ttc']}
                    await websocket.send(json.dumps(config))
                continue 

            if data['type'] == 'frame':
                frame_count += 1
                
                frame = base64_to_image(data['data'])
                if frame is None: continue
                
                if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # YOLO Prediction (Non-blocking)
                def run_yolo_predict():
                    return model.predict(frame, verbose=False, classes=TARGET_CLASSES, conf=USER_CONFIG['detection_confidence'])

                results = await asyncio.to_thread(run_yolo_predict)
                
                # DEBUG LOG
                if frame_count % 30 == 0:
                    count = len(results[0].boxes) if results and results[0].boxes else 0
                    logging.info(f"DEBUG: Frame {frame_count} - YOLO detected {count} objects.")

                detections_output = tracker.update_and_get_ttc(results, current_time, model.names)
                
                if detections_output:
                    current_risk = detections_output[0]['risk_level']
                    smoothed_risk = smoother.update_and_get_risk(current_risk)
                    detections_output[0]['risk'] = smoothed_risk
                    
                    # Log Speed and Risk if active
                    if detections_output[0]['speed_mph'] > 0:
                         logging.info(f"TRACKING: Speed={detections_output[0]['speed_mph']:.1f} MPH, TTC={detections_output[0]['ttc']:.1f}s, Risk={smoothed_risk}")

                response = {
                    'type': 'detection',
                    'timestamp': current_time,
                    'detections': detections_output
                }
                await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosedOK:
        logging.info("Client disconnected (Normal closure)")
    except Exception as e:
        logging.critical(f"FATAL UNHANDLED EXCEPTION: {e}")
        
    finally:
        logging.info("Handler finished.")

# --- Server Initialization ---

async def main_server_runner():
    logging.info("Loading ADAS assets...")
    
    if not os.path.exists(MODEL_PATH):
        logging.critical(f"CRITICAL ERROR: '{MODEL_PATH}' not found.")
        return

    try:
        model = YOLO(MODEL_PATH) 
        logging.info("YOLO model loaded successfully.")
    except Exception as e:
        logging.critical(f"CRITICAL ERROR: Failed to load YOLO model: {e}")
        return
        
    tracker = ObjectTracker()
    smoother = RiskSmoother()

    # FIXED: Handler wrapper that accepts arbitrary arguments (like path) but ignores them
    async def handler_wrapper(websocket, *args):
        await websocket_server_logic(websocket, model, tracker, smoother)

    start_server = websockets.serve(handler_wrapper, "0.0.0.0", 8765)

    logging.info("--- ADAS WebSocket Server Initialized ---")
    logging.info("Websocket Server Running on ws://0.0.0.0:8765")

    async with start_server:
        await asyncio.Future() 

if __name__ == "__main__":
    try:
        asyncio.run(main_server_runner())
    except KeyboardInterrupt:
        logging.info("\nServer manually stopped.")
    except Exception as e:
        logging.critical(f"Server runtime error: {e}")