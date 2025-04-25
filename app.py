from flask import Flask, render_template, request, redirect, url_for, Response, flash, session  
from werkzeug.utils import secure_filename
import os
import socket
import pickle
import cv2
import subprocess
import signal
import threading
import time
from functools import wraps
from flask import after_this_request
# ───────────────────────────────────────────────
# App Configuration
# ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'hef', 'txt', 'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ───────────────────────────────────────────────
# Global Variables
# ───────────────────────────────────────────────
latest_frame = None
current_process = None
stream_active = False
current_mode = None

# ───────────────────────────────────────────────
# UDP Socket Setup
# ───────────────────────────────────────────────
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("0.0.0.0", 9999))
server_socket.settimeout(1.0)

# ───────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def receive_frames():
    global latest_frame
    while True:
        try:
            packet, _ = server_socket.recvfrom(65536)
            buffer = pickle.loads(packet)
            latest_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Frame receive error: {e}")
            break

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None and stream_active:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

def run_detection(model_path, input_source, label_path):
    global current_process
    home_dir = os.path.expanduser("~")
    app_dir = os.path.join(home_dir, "Hailo-AI-Lab")

    if not os.path.exists(app_dir):
        raise FileNotFoundError(f"Directory not found: {app_dir}")

    base_cmd = f"cd {app_dir} && source env/bin/activate && cd AI-LAB && "
    
    if input_source.lower() == 'camera':
        detection_cmd = f"python3 client_object_detection.py -n {model_path} -i camera -l {label_path}"
    else:
        detection_cmd = f"python3 client_object_detection.py -n {model_path} -i {input_source} -l {label_path}"
    
    full_cmd = base_cmd + detection_cmd

    current_process = subprocess.Popen(
        full_cmd,
        shell=True,
        executable='/bin/bash',
        preexec_fn=os.setsid
    )
    
def run_pose_estimation(model_path, input_source):
    global current_process
    home_dir = os.path.expanduser("~")
    app_dir = os.path.join(home_dir, "Hailo-AI-Lab")

    if not os.path.exists(app_dir):
        raise FileNotFoundError(f"Directory not found: {app_dir}")

    base_cmd = f"cd {app_dir} && source env/bin/activate && cd AI-LAB && "

    if input_source.lower() == 'camera':
        detection_cmd = f"python3 client_pose_estimation.py -n {model_path} -i 0"
    else:
        detection_cmd = f"python3 client_pose_estimation.py -n {model_path} -i {input_source}"

    full_cmd = base_cmd + detection_cmd

    current_process = subprocess.Popen(
        full_cmd,
        shell=True,
        executable='/bin/bash',
        preexec_fn=os.setsid
    )

def clear_flash(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        @after_this_request
        def clear_flash_messages(response):
            # This will run after the request is complete
            if '_flashes' in session:
                session.pop('_flashes')
            return response
        return f(*args, **kwargs)
    return decorated_function

# ───────────────────────────────────────────────
# Routes
# ───────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('welcome.html')


@app.route('/object-detection', methods=['GET', 'POST'])
@clear_flash
def object_detection():
    global current_mode, stream_active

    current_mode = 'object'
    stream_active = True

    if request.method == 'POST':
        model_file = request.files.get('model')
        label_file = request.files.get('label')
        input_type = request.form.get('input_type')
        mp4_file = request.files.get('mp4_file')

        # Validation
        if not model_file or not label_file:
            flash("Model and Label files are required.", "danger")
            return redirect(request.url)

        if not allowed_file(model_file.filename) or not allowed_file(label_file.filename):
            flash("Invalid file type.", "danger")
            return redirect(request.url)

        # Save uploaded files
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
        model_file.save(model_path)

        label_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(label_file.filename))
        label_file.save(label_path)

        # Input source
        input_source = 'camera'
        if input_type == 'mp4' and mp4_file and allowed_file(mp4_file.filename):
            input_source = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(mp4_file.filename))
            mp4_file.save(input_source)

        # Run detection
        run_detection(model_path, input_source, label_path)
        flash("Detection started successfully!", "success")
        return redirect(url_for('stream'))

    return render_template("object-detection.html")


@app.route('/pose-estimation', methods=['GET', 'POST'])
@clear_flash
def pose_estimation():
    global current_mode, stream_active

    current_mode = 'pose'
    stream_active = True

    if request.method == 'POST':
        model_file = request.files.get('model')
        input_type = request.form.get('input_type')
        mp4_file = request.files.get('mp4_file')

        if not model_file:
            flash("Model file is required.", "danger")
            return redirect(request.url)

        if not allowed_file(model_file.filename):
            flash("Invalid model file type.", "danger")
            return redirect(request.url)

        # Save the model
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
        model_file.save(model_path)

        # Set input source
        input_source = 'camera'
        if input_type == 'mp4' and mp4_file and allowed_file(mp4_file.filename):
            input_source = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(mp4_file.filename))
            mp4_file.save(input_source)

        # Run pose estimation
        run_pose_estimation(model_path, input_source)
        flash("Pose estimation started!", "success")
        return redirect(url_for('stream'))

    return render_template("pose-estimation.html")

def run_face_detection():
    global current_process
    home_dir = os.path.expanduser("~")
    app_dir = os.path.join(home_dir, "Hailo-AI-Lab")

    if not os.path.exists(app_dir):
        raise FileNotFoundError(f"Directory not found: {app_dir}")

    base_cmd = f"cd {app_dir} && source env/bin/activate && cd AI-LAB && "
    detection_cmd = "python3 client_face_hand.py -d face"
    full_cmd = base_cmd + detection_cmd

    current_process = subprocess.Popen(
        full_cmd,
        shell=True,
        executable='/bin/bash',
        preexec_fn=os.setsid
    )

def run_hand_detection():
    global current_process
    home_dir = os.path.expanduser("~")
    app_dir = os.path.join(home_dir, "Hailo-AI-Lab")

    if not os.path.exists(app_dir):
        raise FileNotFoundError(f"Directory not found: {app_dir}")

    base_cmd = f"cd {app_dir} && source env/bin/activate && cd AI-LAB && "
    detection_cmd = "python3 client_face_hand.py -d hand"
    full_cmd = base_cmd + detection_cmd

    current_process = subprocess.Popen(
        full_cmd,
        shell=True,
        executable='/bin/bash',
        preexec_fn=os.setsid
    )

'''
@app.route('/stop')
def stop():
    global current_process, stream_active

    stream_active = False
    if current_process:
        os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
        current_process = None
        flash("Detection stopped.", "info")

    return redirect(url_for('object_detection'))
'''


@app.route('/stream')
def stream():
    return render_template('stream.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global current_process, stream_active, current_mode
    session.pop('_flashes', None)

    stream_active = False
    if current_process:
        os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
        current_process = None
        flash("Detection stopped.", "info")

    # Store the redirect target in session to handle the flash properly
    if current_mode == 'object':
        session['redirect_target'] = 'object_detection'
    elif current_mode == 'pose':
        session['redirect_target'] = 'pose_estimation'
    elif current_mode == 'face':
        session['redirect_target'] = 'face_detection'
    elif current_mode == 'hand':
        session['redirect_target'] = 'hand_detection'
    else:
        session['redirect_target'] = 'welcome'
        
    return redirect(url_for(session['redirect_target']))

@app.route('/face-detection', methods=['GET', 'POST'])
def face_detection():
    global current_mode
    current_mode = 'face'
    return render_template('face_detection.html')

@app.route('/start_face_detection', methods=['POST'])
@clear_flash
def start_face_detection():
    global stream_active
    
    # Always use camera - simplified version
    run_face_detection()
    stream_active = True
    flash("Hand detection started!", "success") 
    return redirect(url_for('stream'))

@app.route('/hand-detection', methods=['GET', 'POST'])
@clear_flash
def hand_detection():
    global current_mode
    current_mode = 'hand'
    return render_template('hand_detection.html')

@app.route('/start_hand_detection', methods=['POST'])
def start_hand_detection():
    global stream_active
    
    # Always use camera - simplified version
    run_hand_detection()
    stream_active = True
    flash("Face detection started!", "success")
    return redirect(url_for('stream'))


# ───────────────────────────────────────────────
# Start Frame Receiver Thread on App Start
# ───────────────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=receive_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True)

