from flask import Flask, request, render_template, redirect, url_for
import logging
import os
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
USERNAME = 'admin'
PASSWORD = 'password123'

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

# Add logging of IP address on request
@app.before_request
def log_request_info():
    logger.info(f"Request from {request.remote_addr} to {request.path}")

# Start the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
