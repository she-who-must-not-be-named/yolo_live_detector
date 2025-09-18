from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load your custom-trained YOLOv8 model
# Make sure 'best.pt' is in the same folder as this script
model = YOLO('best.pt')

# Initialize video capture from the default webcam (index 0)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Read a frame from the camera
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection on the frame
            results = model(frame)

            # Get the annotated frame with bounding boxes
            annotated_frame = results[0].plot()

            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()

            # Yield the frame in the multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the 'src' attribute of an 'img' tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)