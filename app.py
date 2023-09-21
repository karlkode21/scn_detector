
# import cv2
import subprocess
import os
from os import listdir
from werkzeug.utils import secure_filename
from flask import Flask,request,render_template

UPLOAD_FOLDER = 'mysite/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def make_detection():
    script_path = "/home/jango23/mysite/app_detecta.py"
    subprocess.run(["python", script_path])

holding_folder = '/home/jango23/mysite/static/uploads/'

# Function to delete existing files
def remove_files():
    for item in listdir(holding_folder):
        os.remove(holding_folder + item)

# Creating a dummy function
def dummy():
    return None

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/detector',methods=['POST'])
def detect():

    try:
        remove_files()

    except:
        dummy()

    finally:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # img = cv2.imread(UPLOAD_FOLDER+'/'+filename)

            # Performing Object detection
            make_detection()

            # sketch_img_name = filename.split('.')[0]+"_sketch.jpg"
            sketch_img_name = filename.split('.')[0]+"_detected.jpg"
            # _ = cv2.imwrite(UPLOAD_FOLDER+'/'+sketch_img_name, sketch_img)
            return render_template('home.html',org_img_name=filename,sketch_img_name=sketch_img_name)


if __name__ == '__main__':
    app.run(debug=True)


