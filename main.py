import cv2
import numpy as np
import mediapipe as mp

class PhysicsBasedActionRecognition:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.face_mesh = self.mp_face.FaceMesh(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7,
            max_num_faces=1
        )
        
        # Physics Constants
        self.shadow_threshold = 0.65  
        self.distance_threshold = 12.0
        
        # Light Source Stabilization vars
        self.stable_light_pos = np.array([0, 0], dtype=np.float32)
        self.initialized_light = False

    def detect_light_source(self, frame, face_landmarks):
        """Robust Light Detection: Ignores face, uses heavy blur & smoothing."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Mask out the Face (Paint it black to ignore skin reflections)
        if face_landmarks:
            face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            points = []
            for idx in face_oval:
                pt = face_landmarks.landmark[idx]
                points.append([int(pt.x * w), int(pt.y * h)])
            
            if len(points) > 0:
                mask = np.ones((h, w), dtype=np.uint8) * 255
                cv2.fillPoly(mask, [np.array(points)], 0)
                gray = cv2.bitwise_and(gray, gray, mask=mask)

        # 2. Heavy Blur & Find Max
        gray = cv2.GaussianBlur(gray, (91, 91), 0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        new_pos = np.array(max_loc, dtype=np.float32)
        
        # 3. Exponential Smoothing
        alpha = 0.05 
        if not self.initialized_light:
            self.stable_light_pos = new_pos
            self.initialized_light = True
        else:
            self.stable_light_pos = (self.stable_light_pos * (1 - alpha)) + (new_pos * alpha)
            
        return self.stable_light_pos.astype(int)

    def get_hand_center(self, hand_landmarks):
        indices = [0, 5, 17]
        x_sum = sum([hand_landmarks.landmark[i].x for i in indices])
        y_sum = sum([hand_landmarks.landmark[i].y for i in indices])
        return np.array([x_sum / 3, y_sum / 3])

    def get_mouth_position(self, face_landmarks):
        up = face_landmarks.landmark[13]
        down = face_landmarks.landmark[14]
        return np.array([(up.x + down.x)/2, (up.y + down.y)/2])

    def detect_hand_shadow_on_face(self, frame, hand_landmarks, face_landmarks):
        """Robust ROI-based shadow detection."""
        h_img, w_img, _ = frame.shape
        
        # Get Face Bounding Box
        x_coords = [lm.x for lm in face_landmarks.landmark]
        y_coords = [lm.y for lm in face_landmarks.landmark]
        x_min, x_max = int(min(x_coords) * w_img), int(max(x_coords) * w_img)
        y_min, y_max = int(min(y_coords) * h_img), int(max(y_coords) * h_img)
        
        # Padding
        pad = 20
        x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
        x_max, y_max = min(w_img, x_max + pad), min(h_img, y_max + pad)
        
        face_roi = frame[y_min:y_max, x_min:x_max]
        if face_roi.size == 0: return np.zeros((h_img, w_img), dtype=np.uint8)

        # Shadow Logic (LAB Dark Detection)
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        thresh_val = np.mean(l_channel) * self.shadow_threshold
        _, roi_mask = cv2.threshold(l_channel, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Cleanup
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        
        # Place back in full frame
        full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        full_mask[y_min:y_max, x_min:x_max] = roi_mask
        return full_mask

    def calculate_depth(self, shadow_mask, hand_landmarks, face_landmarks):
        if not hand_landmarks or not face_landmarks: return None, 0
        
        shadow_area = np.count_nonzero(shadow_mask)
        h_center = self.get_hand_center(hand_landmarks)
        m_pos = self.get_mouth_position(face_landmarks)
        geo_dist = np.linalg.norm(h_center - m_pos) * 100 
        
        area_factor = 0
        if shadow_area > 0:
            area_factor = 1000.0 / (shadow_area ** 0.5)
            
        final_depth = (geo_dist * 0.8) + (area_factor * 0.2)
        return final_depth, shadow_area

    def classify_action(self, depth):
        if depth is None: return "No Detection"
        if depth < self.distance_threshold: return "ACTION: EATING / TOUCHING"
        return "IDLE"

    def draw_overlays(self, frame, hand_lm, face_lm, light, shadow_mask, depth, action):
        h, w, _ = frame.shape
        
        # Draw Face Box (Blue)
        if face_lm:
            x_v = [lm.x for lm in face_lm.landmark]
            y_v = [lm.y for lm in face_lm.landmark]
            cv2.rectangle(frame, (int(min(x_v)*w), int(min(y_v)*h)), (int(max(x_v)*w), int(max(y_v)*h)), (255,0,0), 2)
            cv2.putText(frame, "Face", (int(min(x_v)*w), int(min(y_v)*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # Draw Hand Box (Green)
        if hand_lm:
            x_v = [lm.x for lm in hand_lm.landmark]
            y_v = [lm.y for lm in hand_lm.landmark]
            cv2.rectangle(frame, (int(min(x_v)*w), int(min(y_v)*h)), (int(max(x_v)*w), int(max(y_v)*h)), (0,255,0), 2)
            cv2.putText(frame, "Hand", (int(min(x_v)*w), int(min(y_v)*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Draw Shadow Tint
        if np.count_nonzero(shadow_mask) > 0:
            red_layer = np.zeros_like(frame)
            red_layer[:, :, 2] = 255
            frame[shadow_mask > 0] = cv2.addWeighted(frame[shadow_mask > 0], 0.7, red_layer[shadow_mask > 0], 0.3, 0)

        # Info & Light
        color = (0,0,255) if "EATING" in action else (0,255,0)
        cv2.putText(frame, f"Depth: {depth:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, action, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.circle(frame, tuple(light), 15, (0, 255, 255), -1)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = self.hands.process(rgb_frame)
        face_res = self.face_mesh.process(rgb_frame)
        
        light_source = np.array([frame.shape[1]//2, 0])
        matrix_view = np.zeros((100, 100), dtype=np.uint8)

        if hand_res.multi_hand_landmarks and face_res.multi_face_landmarks:
            hand_lm = hand_res.multi_hand_landmarks[0]
            face_lm = face_res.multi_face_landmarks[0]
            
            light_source = self.detect_light_source(frame, face_lm)
            shadow_mask = self.detect_hand_shadow_on_face(frame, hand_lm, face_lm)
            depth, _ = self.calculate_depth(shadow_mask, hand_lm, face_lm)
            action = self.classify_action(depth)
            
            self.draw_overlays(frame, hand_lm, face_lm, light_source, shadow_mask, depth, action)
            
            if np.count_nonzero(shadow_mask) > 0:
                matrix_view = cv2.resize(shadow_mask, (200, 200))
                matrix_view = cv2.applyColorMap(matrix_view, cv2.COLORMAP_JET)

        return frame, matrix_view

def main():
    recognizer = PhysicsBasedActionRecognition()
    cap = cv2.VideoCapture(0) # Change to 1 for external cam
    
    print("System Started. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        main_view, matrix_view = recognizer.process_frame(frame)
        
        cv2.imshow('Physics Action Recognition', main_view)
        if matrix_view.shape[0] > 0 and len(matrix_view.shape)==3:
            cv2.imshow('Shadow Heatmap', matrix_view)
            
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()