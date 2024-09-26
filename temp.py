import cv2
import numpy as np

cap = cv2.VideoCapture(0)

min_area = 100
max_area = 5000
max_contours = 5

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:max_contours]

    shape_count = {
        "Triangle": 0,
        "Square": 0,
        "Rectangle": 0,
        "Circle": 0
    }

    for contour in filtered_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        shape = None
        x, y, w, h = cv2.boundingRect(contour)
        
        if len(approx) == 3:
            shape = "Triangle"
        
        elif len(approx) == 4:
            aspectRatio = w / float(h)
            if 0.95 <= aspectRatio <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        
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
            cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    count_text = f"Triangles: {shape_count['Triangle']} | Squares: {shape_count['Square']} | Rectangles: {shape_count['Rectangle']} | Circles: {shape_count['Circle']}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.imshow("Shape Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()
