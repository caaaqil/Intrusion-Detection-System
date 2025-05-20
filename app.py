from flask import Flask, render_template, request, jsonify, session, url_for, redirect, flash
from joblib import load
import mysql.connector
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta, datetime
import os
from werkzeug.utils import secure_filename
import pickle
import time

# Initialize Flask app and explicitly set template_folder to the current directory
app = Flask(__name__, template_folder='.')
app.secret_key = os.urandom(24)  # Generate a random secret key for sessions

# File upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load your models (adjust path if necessary)
try:
    model = load('models/xgboost_best_model.pkl')
    print("Successfully loaded model")
except FileNotFoundError as e:
    print(f"Error: Could not find model file: {str(e)}")
    raise
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Define the feature mapping globally
feature_mapping = {
    'duration': 0,
    'protocol_type': 1,
    'service': 2,
    'flag': 3,
    'src_bytes': 4,
    'dst_bytes': 5,
    'land': 6,
    'wrong_fragment': 7,
    'urgent': 8,
    'hot': 9,
    'num_failed_logins': 10,
    'logged_in': 11,
    'num_compromised': 12,
    'root_shell': 13,
    'su_attempted': 14,
    'num_root': 15,
    'num_file_creations': 16,
    'num_shells': 17,
    'num_access_files': 18,
    'num_outbound_cmds': 19,
    'is_host_login': 20,
    'is_guest_login': 21,
    'count': 22,
    'srv_count': 23,
    'serror_rate': 24,
    'srv_serror_rate': 25,
    'rerror_rate': 26,
    'srv_rerror_rate': 27,
    'same_srv_rate': 28,
    'diff_srv_rate': 29,
    'srv_diff_host_rate': 30,
    'dst_host_count': 31,
    'dst_host_srv_count': 32,
    'dst_host_same_srv_rate': 33,
    'dst_host_diff_srv_rate': 34,
    'dst_host_same_src_port_rate': 35,
    'dst_host_srv_diff_host_rate': 36,
    'dst_host_serror_rate': 37,
    'dst_host_srv_serror_rate': 38,
    'dst_host_rerror_rate': 39,
    'dst_host_srv_rerror_rate': 40
}

# Define categorical feature mappings (updated to match LabelEncoder output)
protocol_type_mapping = {
    'i c m p': 0,
    't c p': 1,
    'u d p': 2
}

flag_mapping = {
    'OTH': 0,
    'REJ': 1,
    'RSTO': 2,
    'RSTOS0': 3,
    'RSTR': 4,
    'S0': 5,
    'S1': 6,
    'S2': 7,
    'S3': 8,
    'SF': 9,
    'SH': 10
}

service_mapping = {
    'IRC': 0,
    'X11': 1,
    'Z39_50': 2,
    'auth': 3,
    'bgp': 4,
    'courier': 5,
    'csnet_ns': 6,
    'ctf': 7,
    'daytime': 8,
    'discard': 9,
    'domain': 10,
    'domain_u': 11,
    'echo': 12,
    'eco_i': 13,
    'ecr_i': 14,
    'efs': 15,
    'exec': 16,
    'finger': 17,
    'ftp': 18,
    'ftp_data': 19,
    'gopher': 20,
    'hostnames': 21,
    'http': 22,
    'http_443': 23,
    'http_8001': 24,
    'imap4': 25,
    'iso_tsap': 26,
    'klogin': 27,
    'kshell': 28,
    'ldap': 29,
    'link': 30,
    'login': 31,
    'mtp': 32,
    'name': 33,
    'netbios_dgm': 34,
    'netbios_ns': 35,
    'netbios_ssn': 36,
    'netstat': 37,
    'nnsp': 38,
    'nntp': 39,
    'ntp_u': 40,
    'other': 41,
    'pm_dump': 42,
    'pop_2': 43,
    'pop_3': 44,
    'printer': 45,
    'private': 46,
    'red_i': 47,
    'remote_job': 48,
    'rje': 49,
    'shell': 50,
    'smtp': 51,
    'sql_net': 52,
    'ssh': 53,
    'sunrpc': 54,
    'supdup': 55,
    'systat': 56,
    'telnet': 57,
    'tim_i': 58,
    'time': 59,
    'urh_i': 60,
    'urp_i': 61,
    'uucp': 62,
    'uucp_path': 63,
    'vmnet': 64,
    'whois': 65
}

# Print detailed model information
print("\n=== Detailed Model Information ===")
try:
    print(f"Model type: {type(model)}")
    if hasattr(model, 'classes_'):
        print(f"Model classes: {model.classes_}")
        print(f"Class mapping: Class 0 = Normal, Class 1 = Anomaly")
    if hasattr(model, 'n_classes_'):
        print(f"Number of classes: {model.n_classes_}")
    if hasattr(model, 'n_features_in_'):
        print(
            f"Number of features model was trained with: {model.n_features_in_}")
except Exception as e:
    print(f"Error getting model info: {str(e)}")
print("================================\n")

# Test prediction on known normal sample
print("Testing prediction on known normal sample...")
# Initialize with 40 features
test_normal = np.zeros(40)
# Basic Connection Features
test_normal[0] = 2         # duration: 2 seconds (short duration)
test_normal[1] = 0         # protocol_type: TCP (0)
test_normal[2] = 0         # service: HTTP (0)
test_normal[3] = 0         # flag: SF (normal connection establishment)
test_normal[4] = 1024      # src_bytes: 1KB sent
test_normal[5] = 2048      # dst_bytes: 2KB received
test_normal[6] = 0         # land: no land attack
test_normal[7] = 0         # wrong_fragment: no fragmentation errors
test_normal[8] = 0         # urgent: no urgent packets
test_normal[9] = 0         # hot: no hot indicators
test_normal[10] = 0        # num_failed_logins: no failed logins
test_normal[11] = 1        # logged_in: successfully logged in
test_normal[12] = 0        # num_compromised: no compromised conditions
test_normal[13] = 0        # root_shell: no root shell obtained
test_normal[14] = 0        # su_attempted: no su attempts
test_normal[15] = 0        # num_root: no root accesses
test_normal[16] = 0        # num_file_creations: no file creations
test_normal[17] = 0        # num_shells: no shells opened
test_normal[18] = 0        # num_access_files: no access to sensitive files
test_normal[19] = 0        # num_outbound_cmds: no outbound commands
test_normal[20] = 0        # is_host_login: no host login
test_normal[21] = 0        # is_guest_login: no guest login
test_normal[22] = 4        # count: 4 connections to same host
test_normal[23] = 4        # srv_count: 4 connections to same service
test_normal[24] = 0.0      # serror_rate: no SYN errors
test_normal[25] = 0.0      # srv_serror_rate: no SYN errors to same service
test_normal[26] = 0.0      # rerror_rate: no REJ errors
test_normal[27] = 0.0      # srv_rerror_rate: no REJ errors to same service
test_normal[28] = 1.0      # same_srv_rate: all connections to same service
# diff_srv_rate: no connections to different services
test_normal[29] = 0.0
# srv_diff_host_rate: no connections to different hosts
test_normal[30] = 0.0
test_normal[31] = 80       # dst_host_count: 80 connections to same host
test_normal[32] = 80       # dst_host_srv_count: 80 connections to same service
# dst_host_same_srv_rate: all connections to same service
test_normal[33] = 1.0
test_normal[34] = 0.0      # dst_host_diff_srv_rate: no different services
test_normal[35] = 0.2      # dst_host_same_src_port_rate: 20% same source port
test_normal[36] = 0.0      # dst_host_srv_diff_host_rate: no different hosts
test_normal[37] = 0.0      # dst_host_serror_rate: no SYN errors
test_normal[38] = 0.0      # dst_host_srv_serror_rate: no service SYN errors
test_normal[39] = 0.0      # dst_host_rerror_rate: no REJ errors

print(f"Shape of test_normal: {test_normal.shape}")
print(f"Number of features in test_normal: {len(test_normal)}")

try:
    # Make prediction on test sample
    test_pred = model.predict(test_normal.reshape(1, -1))
    test_prob = model.predict_proba(test_normal.reshape(1, -1))[0]
    print(f"Test prediction result: {test_pred[0]}")
    print(f"Test prediction probabilities: {test_prob}")
except Exception as e:
    print(f"Error during prediction: {str(e)}")
    print(
        f"Expected features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'unknown'}")
print("================================\n")

# MySQL configurations for initial connection (without database)
mysql_config_init = {
    'host': 'localhost',
    'user': 'root',
    'password': ''
}

# Full MySQL configuration (with database)
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'intrusion_detection_system'
}


def get_db_connection():
    """Create and return a new database connection"""
    return mysql.connector.connect(**mysql_config)

# Initialize database and tables


def init_database():
    try:
        # Connect without database first
        conn = mysql.connector.connect(**mysql_config_init)
        cursor = conn.cursor()

        # Create database if it doesn't exist
        cursor.execute(
            "CREATE DATABASE IF NOT EXISTS intrusion_detection_system")
        cursor.execute("USE intrusion_detection_system")

        # Create users table FIRST
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                role VARCHAR(20) DEFAULT 'user',
                profile_image VARCHAR(255) DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL
            )
        """)

        # Then create detections table with foreign key
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INT AUTO_INCREMENT PRIMARY KEY,
                prediction VARCHAR(255) NOT NULL,
                confidence FLOAT NOT NULL,
                user_id INT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("Database and tables initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise e


# Initialize database on startup
init_database()

# Home page


@app.route('/')
def index():
    return render_template('index.html')

# Admin page


@app.route('/admin')
def admin():
    return render_template('admin.html')

# Simple Detection page


@app.route('/result')
def result():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    return render_template("result.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Check if user is logged in
    if not session.get('user_id'):
        return render_template("predict.html")

    conn = None
    cursor = None
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()

        if request.method == 'POST':
            # Get the input values from the form
            form_data = request.form.to_dict()

            print("\n=== Processing New Prediction ===")
            print(f"Input data: {form_data}")

            # Initialize a zero array for 41 features
            features = np.zeros(41)

            # Process categorical features with correct mapping
            # protocol_type: convert to match label encoder keys
            if 'protocol_type' in form_data:
                protocol = form_data['protocol_type'].replace(' ', '').lower()
                if protocol == 'icmp':
                    protocol_key = 'i c m p'
                elif protocol == 'tcp':
                    protocol_key = 't c p'
                elif protocol == 'udp':
                    protocol_key = 'u d p'
                else:
                    protocol_key = 'i c m p'  # default or handle error
                features[feature_mapping['protocol_type']] = protocol_type_mapping[protocol_key]

            if 'service' in form_data:
                service = form_data['service']
                features[feature_mapping['service']] = service_mapping.get(service, service_mapping['other'])

            if 'flag' in form_data:
                flag = form_data['flag']
                features[feature_mapping['flag']] = flag_mapping.get(flag, flag_mapping['OTH'])

            # Process numerical features
            for field, index in feature_mapping.items():
                if field not in ['protocol_type', 'service', 'flag']:
                    value = form_data.get(field, '0')
                    try:
                        features[index] = float(value)
                    except ValueError:
                        return render_template("predict.html",
                                               error=f"Invalid value for {field}. Please enter a valid number.")

            print("\nFeature values:")
            for field, index in feature_mapping.items():
                print(f"{field}: {features[index]}")

            # Make prediction
            prediction = model.predict(features.reshape(1, -1))[0]
            probabilities = model.predict_proba(features.reshape(1, -1))[0]
            confidence = float(max(probabilities))
            prediction_str = 'normal' if prediction == 0 else 'anomaly'

            print(f"\nPrediction: {prediction_str}")
            print(f"Confidence: {confidence}")

            # Save to database
            cursor.execute(
                "INSERT INTO detections (prediction, confidence, user_id) VALUES (%s, %s, %s)",
                (prediction_str, confidence, session['user_id'])
            )
            conn.commit()

            # Store prediction results in session for the result page
            session['last_prediction'] = {
                'prediction': prediction_str,
                'confidence': f"{confidence:.2%}"
            }

            return redirect(url_for('result'))

        return render_template("predict.html")
    except Exception as e:
        print(f"Database error in predict: {str(e)}")
        return render_template("predict.html", error="An error occurred while processing your request.")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route('/statistics')
def statistics():
    conn = None
    cursor = None
    try:
        # Initialize default values
        stat_dict = {'normal': 0, 'anomaly': 0}
        total = 0
        user_stats = []
        user_count = 0

        # Get database connection
        conn = get_db_connection()
        # Use dictionary cursor for easier row access
        cursor = conn.cursor(dictionary=True)

        # Get statistics from database
        if session.get('role') == 'admin':
            # Get total user count
            cursor.execute("SELECT COUNT(*) as count FROM users")
            user_count = cursor.fetchone()['count']

            # Get per-user statistics
            cursor.execute("""
                SELECT u.username, COUNT(d.id) as detection_count
                FROM users u
                LEFT JOIN detections d ON u.id = d.user_id
                GROUP BY u.id, u.username
                ORDER BY detection_count DESC
            """)
            user_stats = cursor.fetchall()

            # Get overall statistics
            cursor.execute("""
                SELECT prediction, COUNT(*) as count 
                FROM detections 
                GROUP BY prediction
            """)
        else:
            # Get user-specific statistics
            cursor.execute("""
                SELECT prediction, COUNT(*) as count 
                FROM detections 
                WHERE user_id = %s
                GROUP BY prediction
            """, (session.get('user_id'),))

        stats = cursor.fetchall()

        # Update stats if we have data
        if stats:
            total = sum(row['count'] for row in stats)
            for row in stats:
                stat_dict[row['prediction']] = row['count']

        return render_template(
            "statistics.html",
            stats=stat_dict,
            total=total,
            is_admin=session.get('role') == 'admin',
            user_stats=user_stats,
            user_count=user_count
        )

    except Exception as e:
        print(f"Database error in statistics: {str(e)}")
        return render_template(
            "statistics.html",
            stats={'normal': 0, 'anomaly': 0},
            total=0,
            is_admin=session.get('role') == 'admin',
            user_stats=[],
            user_count=0
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route('/admin/users', methods=['GET', 'POST', 'DELETE'])
def admin_users():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Unauthorized access', 'error')
        return redirect(url_for('index'))

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        if request.method == 'POST':
            # Handle PATCH method for role updates
            if request.form.get('_method') == 'PATCH':
                user_id = request.form.get('user_id')
                new_role = request.form.get('role')

                # Prevent admin from changing their own role
                if int(user_id) == session['user_id']:
                    flash('You cannot change your own role', 'error')
                else:
                    try:
                        cursor.execute(
                            "UPDATE users SET role = %s WHERE id = %s",
                            (new_role, user_id)
                        )
                        conn.commit()
                        flash('User role updated successfully!', 'success')
                    except Exception as e:
                        flash(f'Error updating user role: {str(e)}', 'error')
                        conn.rollback()

            # Handle user creation
            elif '_method' not in request.form:
                username = request.form.get('username')
                email = request.form.get('email')
                password = request.form.get('password')
                full_name = request.form.get('full_name')
                role = request.form.get('role', 'user')

                if all([username, email, password, full_name]):
                    hashed_password = generate_password_hash(password)
                    try:
                        cursor.execute(
                            "INSERT INTO users (username, email, password, full_name, role) VALUES (%s, %s, %s, %s, %s)",
                            (username, email, hashed_password, full_name, role)
                        )
                        conn.commit()
                        flash('User created successfully!', 'success')
                    except Exception as e:
                        flash(f'Error creating user: {str(e)}', 'error')
                        conn.rollback()

            # Handle user deletion
            elif request.form.get('_method') == 'DELETE':
                user_id = request.form.get('user_id')
                # Prevent admin from deleting themselves
                if int(user_id) == session['user_id']:
                    flash('You cannot delete your own account', 'error')
                else:
                    try:
                        # Delete user's detections first
                        cursor.execute(
                            "DELETE FROM detections WHERE user_id = %s", (user_id,))
                        # Then delete the user
                        cursor.execute(
                            "DELETE FROM users WHERE id = %s", (user_id,))
                        conn.commit()
                        flash('User deleted successfully!', 'success')
                    except Exception as e:
                        flash(f'Error deleting user: {str(e)}', 'error')
                        conn.rollback()

        # Get all users except current admin
        cursor.execute("""
            SELECT u.*, COUNT(d.id) as detection_count
            FROM users u
            LEFT JOIN detections d ON u.id = d.user_id
            WHERE u.id != %s
            GROUP BY u.id
            ORDER BY u.created_at DESC
        """, (session['user_id'],))
        users = cursor.fetchall()

        return render_template('admin_users.html', users=users)

    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            full_name = request.form['full_name']

            # Basic validation
            if not all([username, email, password, full_name]):
                return render_template('signup.html', error="All fields are required")

            # Hash the password
            hashed_password = generate_password_hash(password)

            conn = get_db_connection()
            cursor = conn.cursor()

            # Check if username or email already exists
            cursor.execute(
                "SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
            if cursor.fetchone():
                return render_template('signup.html', error="Username or email already exists")

            # Insert new user with default 'user' role
            cursor.execute(
                "INSERT INTO users (username, email, password, full_name, role) VALUES (%s, %s, %s, %s, 'user')",
                (username, email, hashed_password, full_name)
            )
            conn.commit()

            return redirect(url_for('login', message="Registration successful! Please login."))

        except Exception as e:
            print(f"Error in signup: {str(e)}")
            return render_template('signup.html', error="An error occurred during registration")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        conn = None
        cursor = None
        try:
            username = request.form['username']
            password = request.form['password']
            remember = 'remember' in request.form

            if not username or not password:
                return render_template('login.html', error="Username and password are required")

            print(f"Attempting login for user: {username}")  # Debug log

            conn = get_db_connection()
            if not conn:
                print("Failed to establish database connection")  # Debug log
                return render_template('login.html', error="Database connection error")

            cursor = conn.cursor(dictionary=True)

            # Get user
            cursor.execute(
                "SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()

            print(f"User found: {bool(user)}")  # Debug log

            if user and check_password_hash(user['password'], password):
                # Debug log
                print(f"Password check passed for user: {username}")

                try:
                    # Update last login
                    cursor.execute(
                        "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                        (user['id'],)
                    )
                    conn.commit()

                    # Create session with all necessary user data
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    session['role'] = user['role']
                    session['full_name'] = user['full_name']
                    session['email'] = user['email']
                    session['profile_image'] = user['profile_image']

                    if remember:
                        session.permanent = True
                        app.permanent_session_lifetime = timedelta(days=7)

                    # Debug log
                    print(f"Login successful for user: {username}")
                    return redirect(url_for('index'))

                except Exception as e:
                    print(f"Error updating last login: {str(e)}")  # Debug log
                    conn.rollback()
                    return render_template('login.html', error="Error updating login timestamp")
            else:
                print(f"Invalid credentials for user: {username}")  # Debug log
                return render_template('login.html', error="Invalid username or password")

        except mysql.connector.Error as e:
            print(f"MySQL Error: {str(e)}")  # Debug log
            return render_template('login.html', error=f"Database error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error in login: {str(e)}")  # Debug log
            return render_template('login.html', error="An unexpected error occurred")
        finally:
            if cursor:
                cursor.close()
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    print(f"Error closing connection: {str(e)}")  # Debug log

    return render_template('login.html', message=request.args.get('message'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login', message="You have been logged out"))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get user data
        cursor.execute("""
            SELECT u.*, 
                   COUNT(d.id) as predictions_count,
                   MAX(d.timestamp) as last_activity
            FROM users u
            LEFT JOIN detections d ON u.id = d.user_id
            WHERE u.id = %s
            GROUP BY u.id
        """, (session['user_id'],))

        user = cursor.fetchone()

        if not user:
            session.clear()
            return redirect(url_for('login'))

        if request.method == 'POST':
            try:
                # Handle profile image upload
                if 'profile_image' in request.files:
                    file = request.files['profile_image']
                    if file and file.filename and allowed_file(file.filename):
                        # Delete old profile image if it exists
                        if user['profile_image']:
                            old_file_path = os.path.join(
                                app.config['UPLOAD_FOLDER'], user['profile_image'])
                            try:
                                if os.path.exists(old_file_path):
                                    os.remove(old_file_path)
                                    print(
                                        f"Successfully deleted old profile image: {old_file_path}")
                            except Exception as e:
                                print(
                                    f"Error deleting old profile image: {str(e)}")

                        # Generate secure filename and save
                        filename = secure_filename(file.filename)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                        unique_filename = timestamp + filename

                        # Ensure upload folder exists
                        if not os.path.exists(app.config['UPLOAD_FOLDER']):
                            os.makedirs(app.config['UPLOAD_FOLDER'])

                        # Save the file
                        file_path = os.path.join(
                            app.config['UPLOAD_FOLDER'], unique_filename)
                        file.save(file_path)
                        print(
                            f"Successfully saved new profile image: {file_path}")

                        # Update user profile with new image filename
                        cursor.execute(
                            "UPDATE users SET profile_image = %s WHERE id = %s",
                            (unique_filename, session['user_id'])
                        )

                        # Update session with new profile image
                        session['profile_image'] = unique_filename

                # Update other profile fields
                username = request.form.get('username', user['username'])
                email = request.form.get('email', user['email'])
                full_name = request.form.get('full_name', user['full_name'])

                cursor.execute(
                    "UPDATE users SET username = %s, email = %s, full_name = %s WHERE id = %s",
                    (username, email, full_name, session['user_id'])
                )

                # Update session data
                session['username'] = username
                session['email'] = email
                session['full_name'] = full_name

                conn.commit()
                flash('Profile updated successfully!', 'success')

                # Refresh user data after update
                cursor.execute("""
                    SELECT u.*, 
                           COUNT(d.id) as predictions_count,
                           MAX(d.timestamp) as last_activity
                    FROM users u
                    LEFT JOIN detections d ON u.id = d.user_id
                    WHERE u.id = %s
                    GROUP BY u.id
                """, (session['user_id'],))
                user = cursor.fetchone()

                return redirect(url_for('profile'))

            except Exception as e:
                conn.rollback()
                flash('An error occurred while updating your profile.', 'error')
                print(f"Error updating profile: {str(e)}")

        return render_template('profile.html',
                               user=user,
                               profile_image_url=url_for('static', filename=f'uploads/{user["profile_image"]}') if user['profile_image'] else None)

    except Exception as e:
        print(f"Error in profile route: {str(e)}")
        flash('An error occurred while loading your profile.', 'error')
        return redirect(url_for('login'))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# Add this after creating the Flask app
app.permanent_session_lifetime = timedelta(
    days=7)  # For "remember me" functionality


@app.route('/team')
def team():
    return render_template('team.html')


if __name__ == '__main__':
    app.run(debug=True)
