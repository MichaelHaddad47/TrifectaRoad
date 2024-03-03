import numpy as np
import cv2
from collections import deque, Counter

# Function to get the name of the color based on HSV values
def get_color_name(hsv):
    if (hsv[0] <= 50 or hsv[0] >= 150) and hsv[1] <= 50:  # HSV range for white
        return 'White'
    elif hsv[2] <= 30:  # Value for black
        return 'Black'
    elif (0 <= hsv[0] <= 12 or 160 <= hsv[0] <= 180) and hsv[1] > 70 and hsv[2] > 50:  # Improved range for red
        return 'Red'
    elif 25 <= hsv[0] <= 35:  # Hue range for yellow
        return 'Yellow'
    elif 0 <= hsv[0] <= 180 and 0 <= hsv[1] <= 25:  # Saturation for silver
        return 'Silver'
    else:
        return 'Other'  # Default if color does not match above conditions

# Maximum number of recent color detections to consider
MAX_HISTORY = 10

# Create a dictionary where the key is the car identifier and the value is a deque of recent colors
color_history = {}

# Function to get the most common color from history
def most_common_color(history):
    return Counter(history).most_common(1)[0][0]

# Load the classifiers
videos = ['Video2.mp4', '4K Road traffic video for object detection and tracking.mp4' ]
car_cascade = cv2.CascadeClassifier('cars.xml')

for video_path in videos:
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frames = cap.read()
        if not ret:
            print(f"{video_path}")
            break

        # Convert frame to HSV and apply histogram equalization to improve contrast in V channel
        hsv = cv2.cvtColor(frames, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        frames = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.05, 5)

        for i, (x, y, w, h) in enumerate(cars):
            car_region = frames[y:y + h, x:x + w]
            hsv_car_region = cv2.cvtColor(car_region, cv2.COLOR_BGR2HSV)  # Convert to HSV
            avg_color_per_row = np.average(hsv_car_region, axis=0)
            avg_hsv = np.average(avg_color_per_row, axis=0).astype(int)
            color_name = get_color_name(avg_hsv)

            # Update color history for each car
            if i not in color_history:
                color_history[i] = deque(maxlen=MAX_HISTORY)
            color_history[i].append(color_name)
            most_common = most_common_color(color_history[i])

            # Draw a rectangle around the car and put the most common color name text above the car
            cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frames, f"{most_common} Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Resize and display the small pop-up with the detected car region
            small_popup = cv2.resize(car_region, (150, 150))
            cv2.imshow('Detected Car', small_popup)

            # Create a color histogram
            hsv_channels = cv2.split(hsv_car_region)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR values for blue, green, red
            for j, col in enumerate(colors):
                hist = cv2.calcHist([hsv_channels[j]], [0], None, [256], [0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=150, norm_type=cv2.NORM_MINMAX)
                hist = hist.reshape(-1)
                for k, val in enumerate(hist):
                    cv2.line(small_popup, (k, 150), (k, 150 - int(val)), col, 1)
            cv2.imshow('Color Histogram', small_popup)

        # Display the main frame
        cv2.imshow('Car Detection System', frames)

        frames = cv2.resize(frames, (960, 540))  # Resize to half of 1920x1080

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture for the current video
    cap.release()

# Close all windows
cv2.destroyAllWindows()