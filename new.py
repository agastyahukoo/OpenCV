import cv2
import numpy as np
import time

class ObjectDetectionDebugger:
    def __init__(self):
        self.fullscreen = False
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.total_frames = 0
        self.fps_counter_start = time.time()
        self.last_print_time = time.time()

    def toggle_fullscreen(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.fullscreen:
                cv2.setWindowProperty("Debugger", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                self.fullscreen = False
            else:
                cv2.setWindowProperty("Debugger", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                self.fullscreen = True

    def create_color_mask(self, hsv_frame):
        lower_color1 = np.array([0, 100, 100])
        upper_color1 = np.array([10, 255, 255])
        lower_color2 = np.array([160, 100, 100])
        upper_color2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_frame, lower_color1, upper_color1)
        mask2 = cv2.inRange(hsv_frame, lower_color2, upper_color2)
        return cv2.bitwise_or(mask1, mask2)

    def preprocess_frame_for_contours(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        edges = cv2.Canny(blurred, 30, 150)
        return edges

    def filter_and_sort_contours(self, contours, min_area, max_area, max_contours):
        filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
        return sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:max_contours]

    def detect_shapes_and_draw(self, contours, frame):
        shape_count = {"Triangle": 0, "Square": 0, "Rectangle": 0, "Circle": 0}
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            shape = None
            x, y, w, h = cv2.boundingRect(contour)
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                aspect_ratio = w / float(h)
                shape = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
            elif len(approx) > 4:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter != 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > 0.8:
                        shape = "Circle"
            if shape:
                shape_count[shape] += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return shape_count

    def print_diagnostic_data(self, shape_count):
        current_time = time.strftime("%H:%M:%S")
        diagnostic_msg = (f"[{current_time}] Circle: {shape_count['Circle']}, "
                          f"Squares: {shape_count['Square']}, "
                          f"Rectangle: {shape_count['Rectangle']}, "
                          f"Triangle: {shape_count['Triangle']}")
        print(diagnostic_msg)

    def create_data_section(self, shape_count, fps, resolution, total_frames, width):
        data_section = np.zeros((150, width, 3), dtype=np.uint8)
        text_color = (255, 255, 255)
        shape_info = f"Triangles: {shape_count['Triangle']} | Squares: {shape_count['Square']} | Rectangles: {shape_count['Rectangle']} | Circles: {shape_count['Circle']}"
        fps_info = f"FPS: {fps:.2f}"
        resolution_info = f"Resolution: {resolution[0]}x{resolution[1]}"
        total_frames_info = f"Total Frames Processed: {total_frames}"
        cv2.putText(data_section, shape_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(data_section, fps_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(data_section, resolution_info, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(data_section, total_frames_info, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        return data_section

    def process_frame(self, frame, min_area, max_area, max_contours):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self.create_color_mask(hsv)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        edges = self.preprocess_frame_for_contours(masked_frame)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = self.filter_and_sort_contours(contours, min_area, max_area, max_contours)
        return self.detect_shapes_and_draw(filtered_contours, frame), mask

    def object_detection(self):
        min_area = 100
        max_area = 50000
        max_contours = 10
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            start_time = time.time()
            shape_count, _ = self.process_frame(frame, min_area, max_area, max_contours)
            self.total_frames += 1
            end_time = time.time()
            fps = self.total_frames / (end_time - self.fps_counter_start)
            current_time = time.time()
            if current_time - self.last_print_time >= 1:
                self.print_diagnostic_data(shape_count)
                self.last_print_time = current_time
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()

    def run_debugger_ui(self):
        min_area = 100
        max_area = 50000
        max_contours = 10
        cv2.namedWindow("Debugger")
        cv2.setMouseCallback("Debugger", self.toggle_fullscreen)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            shape_count, mask = self.process_frame(frame, min_area, max_area, max_contours)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined_frame = np.hstack((frame, mask_bgr))
            self.total_frames += 1
            end_time = time.time()
            fps = self.total_frames / (end_time - self.fps_counter_start)
            resolution = (frame.shape[1], frame.shape[0])
            data_section = self.create_data_section(shape_count, fps, resolution, self.total_frames, combined_frame.shape[1])
            final_output = np.vstack((combined_frame, data_section))
            cv2.imshow("Debugger", final_output)
            if cv2.waitKey(1) & 0xFF == ord('d'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    debugger = ObjectDetectionDebugger()
    mode = input("Enter 'd' for Debugger mode or 'r' for Regular object detection: ")
    if mode == 'd':
        debugger.run_debugger_ui()
    elif mode == 'r':
        debugger.object_detection()

if __name__ == "__main__":
    main()
