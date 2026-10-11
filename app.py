import os
import sys
import time
import uuid
import json
import sqlite3
import io
import contextlib
import threading
import shutil

import torch
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory

# Project imports
from restore import restore_image, load_model, diagnose_scene
from feedback import save_feedback
from auto_improve import analyze_and_improve, _load_isp_params

# ── Paths & Setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
FEEDBACK_DB = os.path.join(BASE_DIR, 'feedback_db', 'feedback.sqlite')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Device & Model Initialization ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print(f"[*] Detected device: {DEVICE}")
print("[*] Loading model 'exp11'...")
MODEL = load_model('exp11', DEVICE)
print("[✓] Model loaded and ready.")

# ── Flask Application ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}

# In-memory task tracking
tasks = {}

# ── CORS Headers ──────────────────────────────────────────────────────────────
@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        return app.make_default_options_response()

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# ── Background Worker ─────────────────────────────────────────────────────────
def _run_restore(task_id, input_path):
    try:
        tasks[task_id]['status'] = 'processing'
        img = Image.open(input_path).convert('RGB')
        x = T.ToTensor()(img).unsqueeze(0).to(DEVICE)
        diag = diagnose_scene(x)
        tasks[task_id]['scene_type'] = diag['scene_type']
        tasks[task_id]['diag'] = {k: float(v) if isinstance(v, (float, int)) else v for k, v in diag.items()}
        output = restore_image(MODEL, img, DEVICE)
        out_filename = f'{task_id}_restored.png'
        out_path = os.path.join(UPLOAD_FOLDER, out_filename)
        output.save(out_path)
        tasks[task_id]['output_filename'] = out_filename
        tasks[task_id]['status'] = 'done'
        tasks[task_id]['end_ms'] = time.time() * 1000
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['end_ms'] = time.time() * 1000

# ── API Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/restore', methods=['POST'])
def api_restore():
    # Cleanup: maintain task dict size <= 100
    if len(tasks) > 100:
        oldest_keys = sorted(tasks.keys(), key=lambda k: tasks[k].get('start_ms', 0))[:50]
        for k in oldest_keys:
            tasks.pop(k, None)

    if 'image' not in request.files:
        return jsonify({'error': 'Missing image file in request field "image"'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported file type: .{ext}'}), 400

    task_id = str(uuid.uuid4())
    input_filename = f"{task_id}_input.{ext}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    file.save(input_path)

    tasks[task_id] = {
        'status': 'pending',
        'scene_type': None,
        'diag': None,
        'input_filename': input_filename,
        'output_filename': None,
        'error': None,
        'start_ms': time.time() * 1000,
        'end_ms': None
    }

    thread = threading.Thread(target=_run_restore, args=(task_id, input_path), daemon=True)
    thread.start()

    return jsonify({'task_id': task_id, 'status': 'processing'})

@app.route('/api/status/<task_id>', methods=['GET'])
def api_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = tasks[task_id]
    start_ms = task.get('start_ms')
    end_ms = task.get('end_ms')
    processing_time_ms = None
    if start_ms is not None and end_ms is not None:
        processing_time_ms = round(end_ms - start_ms, 2)
    elif start_ms is not None:
        processing_time_ms = round(time.time() * 1000 - start_ms, 2)

    input_url = f"/uploads/{task['input_filename']}" if task.get('input_filename') else None
    output_url = f"/uploads/{task['output_filename']}" if task.get('output_filename') else None
    status = task.get('status', 'pending')
    if status == 'done':
        status = 'completed'

    return jsonify({
        'status': status,
        'task_id': task_id,
        'scene_type': task.get('scene_type'),
        'diag': task.get('diag'),
        'output_url': output_url,
        'restored_url': output_url,
        'input_url': input_url,
        'processing_time_ms': processing_time_ms,
        'processing_time': round(processing_time_ms / 1000.0, 2) if processing_time_ms else 1.2,
        'error': task.get('error')
    })

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON request'}), 400

    task_id = data.get('task_id')
    rating = data.get('rating')
    note = data.get('note', '')

    if not task_id or task_id not in tasks:
        return jsonify({'error': 'Task ID not found'}), 404

    try:
        rating = int(rating)
        if not (1 <= rating <= 5):
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({'error': 'Rating must be an integer between 1 and 5'}), 400

    task = tasks[task_id]
    input_filename = task.get('input_filename')
    output_filename = task.get('output_filename')
    input_path = os.path.join(UPLOAD_FOLDER, input_filename) if input_filename else ''
    output_path = os.path.join(UPLOAD_FOLDER, output_filename) if output_filename else ''
    scene_type = task.get('scene_type')
    diag = task.get('diag')

    try:
        save_feedback(
            input_path=input_path,
            output_path=output_path,
            rating=rating,
            note=note,
            scene_type=scene_type,
            diag=diag
        )
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _path_to_url(filepath):
    if not filepath or not os.path.exists(filepath):
        return None
    abs_path = os.path.abspath(filepath)
    abs_upload = os.path.abspath(UPLOAD_FOLDER)
    if abs_path.startswith(abs_upload):
        rel = os.path.relpath(abs_path, abs_upload)
        return f"/uploads/{rel}"
    dest_name = "hist_" + os.path.basename(filepath)
    dest_path = os.path.join(UPLOAD_FOLDER, dest_name)
    if not os.path.exists(dest_path):
        try:
            shutil.copy2(filepath, dest_path)
        except Exception:
            return None
    return f"/uploads/{dest_name}"

@app.route('/api/history', methods=['GET'])
def api_history():
    if not os.path.exists(FEEDBACK_DB):
        return jsonify([])

    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, input_path, output_path, scene_type, rating, note
            FROM restoration_feedback
            ORDER BY id DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            inp_url = _path_to_url(row[2])
            out_url = _path_to_url(row[3])
            records.append({
                'id': row[0],
                'task_id': f"hist_{row[0]}",
                'timestamp': row[1],
                'created_at': row[1],
                'input_path': row[2],
                'output_path': row[3],
                'input_url': inp_url,
                'output_url': out_url,
                'restored_url': out_url,
                'scene_type': row[4],
                'rating': row[5],
                'note': row[6],
                'comment': row[6]
            })
        return jsonify(records)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    if not os.path.exists(FEEDBACK_DB):
        return jsonify({
            'total': 0,
            'avg_rating': 0.0,
            'good_count': 0,
            'bad_count': 0,
            'by_scene': []
        })

    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM restoration_feedback")
        total_row = cursor.fetchone()
        total = total_row[0] if total_row else 0

        if total == 0:
            conn.close()
            return jsonify({
                'total': 0,
                'avg_rating': 0.0,
                'good_count': 0,
                'bad_count': 0,
                'by_scene': []
            })

        cursor.execute("SELECT AVG(rating) FROM restoration_feedback")
        avg_row = cursor.fetchone()
        avg_rating = round(float(avg_row[0]), 2) if avg_row and avg_row[0] is not None else 0.0

        cursor.execute("SELECT COUNT(*) FROM restoration_feedback WHERE rating >= 4")
        good_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM restoration_feedback WHERE rating <= 2")
        bad_count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT scene_type, COUNT(*) AS n, AVG(rating) AS avg_r,
                   SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) AS bad_n
            FROM restoration_feedback
            GROUP BY scene_type
            ORDER BY avg_r ASC
        """)
        rows = cursor.fetchall()
        conn.close()

        by_scene = [
            {
                'scene_type': row[0] or 'UNKNOWN',
                'count': row[1],
                'avg_rating': round(float(row[2]), 2) if row[2] is not None else 0.0,
                'bad_count': row[3] if row[3] is not None else 0
            }
            for row in rows
        ]

        return jsonify({
            'total': total,
            'avg_rating': avg_rating,
            'good_count': good_count,
            'bad_count': bad_count,
            'by_scene': by_scene
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auto-improve', methods=['POST'])
def api_auto_improve():
    try:
        data = request.get_json(silent=True) or {}
        scene_type = data.get('scene_type')
        if scene_type == '':
            scene_type = None

        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            analyze_and_improve(scene_type=scene_type, dry_run=False)

        msg = captured_output.getvalue().strip()
        return jsonify({
            'success': True,
            'message': msg if msg else 'Auto-improvement completed.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# ── Main Entry Point ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"\n🚀 DAResUNet Web App starting on http://localhost:{port} (MPS accelerated)")
    app.run(host='0.0.0.0', port=port, debug=False)

