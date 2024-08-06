from flask import Flask, request, render_template, redirect, url_for, send_file
import logging
import os
import csv
from datetime import datetime
from functools import wraps

# Set up Flask application
app = Flask(__name__)

# Log file path
log_file_path = os.path.join('logs', 'app.log')

# Create logs directory if it does not exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logger
logger = logging.getLogger('app_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Sample username and password for authentication
USERNAME = 'administrator'
PASSWORD = 'PasswordWebBIT2024*'

# Authentication decorator
def require_authentication(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != USERNAME or auth.password != PASSWORD:
            return ('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated_function

# Route for displaying logs
@app.route('/')
@require_authentication
def show_logs():
    logs = []
    with open(log_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(" : ", 1)
            timestamp = parts[0]
            message = parts[1]
            logs.append({"timestamp": timestamp, "message": message})
    return render_template('logs.html', logs=logs)

# Route to clear logs
@app.route('/clear_logs', methods=['POST'])
@require_authentication
def clear_logs():
    with open(log_file_path, 'w') as file:
        file.write('')  # Clear the log file content
    logger.info("Log telah dibersihkan.")
    return redirect(url_for('show_logs'))

# Route to download logs as CSV
@app.route('/download_csv')
@require_authentication
def download_csv():
    csv_file_path = os.path.join('logs', 'log.csv')
    logs = []

    with open(log_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(" : ", 1)
            timestamp = parts[0]
            message = parts[1]
            logs.append({"timestamp": timestamp, "message": message})

    # Write logs to CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'message']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for log in logs:
            writer.writerow(log)

    logger.info("Log telah disimpan ke CSV.")
    return send_file(csv_file_path, as_attachment=True)

# Add logging of IP address on request
@app.before_request
def log_request_info():
    logger.info(f"Request dari {request.remote_addr} ke {request.path}")

# Start the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
