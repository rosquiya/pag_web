from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session, flash
import cv2
import os
import base64
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from prepro import process_image  # Asegúrate de tener este archivo con la función process_image

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Ruta a la carpeta de descargas
DOWNLOAD_FOLDER = os.path.join(os.path.expanduser("~"), 'Downloads')
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

# Ruta al archivo de configuración de la cámara
CONFIG_FILE = 'camera_config.json'

# Cargar configuraciones desde el archivo JSON
def load_camera_config():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

camera_config = load_camera_config()

# Configuración de la cámara
camera = cv2.VideoCapture(1)  # Cambia el índice si tienes más de una cámara

def apply_camera_settings(camera, config):
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Desactivar el autoenfoque
    camera.set(cv2.CAP_PROP_BRIGHTNESS, config.get('brightness', 128))
    camera.set(cv2.CAP_PROP_SATURATION, config.get('saturation', 128))
    camera.set(cv2.CAP_PROP_CONTRAST, config.get('contrast', 128))
    camera.set(cv2.CAP_PROP_FOCUS, config.get('focus', 50))
    camera.set(cv2.CAP_PROP_EXPOSURE, -6)  # Ajuste de exposición
    camera.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4800)

apply_camera_settings(camera, camera_config)

# Cargar modelo YOLO
model = YOLO('best.pt')  # Asegúrate de que 'best.pt' esté en la ubicación correcta

# Renderizar la página principal
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user')
def user():
    return render_template('pantalla_usuario.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Usuario o contraseña incorrectos.')
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin' in session:
        camera_config = load_camera_config()  # Cargar la configuración de nuevo al renderizar la página
        return render_template('admin_dashboard.html', camera_config=camera_config)
    else:
        flash('Debe iniciar sesión primero.')
        return redirect(url_for('admin'))

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('index'))

@app.route('/admin/save_camera_config', methods=['POST'])
def save_camera_config():
    new_config = {
        'brightness': int(request.form['brightness']),
        'saturation': int(request.form['saturation']),
        'contrast': int(request.form['contrast']),
        'focus': int(request.form['focus'])
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(new_config, f)
    apply_camera_settings(camera, new_config)
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/update_camera_settings', methods=['POST'])
def update_camera_settings():
    new_settings = request.json
    apply_camera_settings(camera, new_settings)
    return '', 204

@app.route('/capture_image', methods=['POST'])
def capture_image():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_path = os.path.join(DOWNLOAD_FOLDER, f'captured_image_{timestamp}.jpg')

    success, frame = camera.read()
    if not success:
        return jsonify({'success': False, 'message': 'Failed to capture image.'})

    cv2.imwrite(img_path, frame)
    
    with open(img_path, 'rb') as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    return jsonify({'success': True, 'message': 'Image saved successfully.', 'path': img_path, 'image': img_base64})

# Generar flujo de video
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Capturar imagen y realizar inferencia
@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if not success:
        return jsonify({'success': False, 'message': 'Failed to capture image.'})

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_path = os.path.join(DOWNLOAD_FOLDER, f'captured_image_{timestamp}.jpg')
    cv2.imwrite(img_path, frame)

    # Procesar la imagen
    processed_image = process_image(img_path)

    if processed_image is not None:
        results = model(processed_image)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        max_prob_index = np.argmax(probs)

        with open(img_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': img_base64,
            'predictions': {
                'names_dict': names_dict,
                'probs': probs,
                'max_prob': names_dict[max_prob_index]
            }
        })
    else:
        return jsonify({'success': False, 'message': f'Failed to process image: {img_path}'})

if __name__ == "__main__":
    app.run(debug=True)
