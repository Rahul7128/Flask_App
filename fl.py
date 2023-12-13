from flask import Flask,Response,render_template, request,flash,redirect, url_for,jsonify
from werkzeug.utils import secure_filename
from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from functiona import *
import os # To import test image files
import cv2 # To work with opencv images
from PIL import Image # Image submodule to work with pillow images
import pytesseract as pt # pytesseract module
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, validators
import numpy as np
import cv2
from keras.models import load_model
import string

app = Flask(__name__)
UPLOAD_FOLDER = 'Static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(500), nullable=False)

class ContactForm(FlaskForm):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    email = StringField('Email', [validators.Length(min=6, max=50)])
    message = StringField('Message', [validators.Length(min=1, max=500)])
    submit = SubmitField('Submit')
    
# Create the database tables
with app.app_context():
    db.create_all()


class RegistrationForm(FlaskForm):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    password = PasswordField('Password', [validators.DataRequired()])
    submit = SubmitField('Login')

@app.route("/home")
def home():
    return render_template('index copy.html')
@app.route("/know")
def after():
    return render_template('index1.html')

@app.route("/hundgestnum")
def hundgestnum():
    return render_template('tut1.html')
def handgetsurenum(): 
    json_file = open("model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("model.h5")

    colors = [(245,117,16) for _ in range(20)]

    # 1. New detection variables
    sequence = []
    sentence = []
    accuracy = []
    predictions = []
    threshold = 0.8 

    cap = cv2.VideoCapture(0)

    # Set mediapipe model 
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            # Make detections
            cropframe = frame[40:400, 0:300]
            frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
            image, results = mediapipe_detection(cropframe, hands)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try: 
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(np.argmax(res))
                    predictions.append(np.argmax(res))
                
                    if np.unique(predictions[-10:])[0] == np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if str(np.argmax(res)) != sentence[-1]:
                                    sentence.append(str(np.argmax(res)))
                                    accuracy.append(str(res[np.argmax(res)]*100))
                            else:
                                sentence.append(str(np.argmax(res)))
                                accuracy.append(str(res[np.argmax(res)]*100)) 

                    if len(sentence) > 1: 
                        sentence = sentence[-1:]
                        accuracy = accuracy[-1:]

            except Exception as e:
                pass
            
            cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
            cv2.putText(frame, "Output: -" + ' '.join(sentence) + "Acc: -" + ''.join(accuracy), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', frame)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(handgetsurenum(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/Aphges')
def Aphgest():
    return render_template('tut2.html')
def handgetsure():
    json_file = open("model1.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("model1.h5")

    colors = []
    for i in range(0,20):
        colors.append((245,117,16))
    print(len(colors))
    def prob_viz(res, actions, input_frame, colors,threshold):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        return output_frame 
    # 1. New detection variables
    sequence = []
    sentence = []
    accuracy=[]
    predictions = []
    threshold = 0.8 

    cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("https://192.168.43.41:8080/video")
# Set mediapipe model 
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            cropframe=frame[40:400,0:300]
            # print(frame.shape)
            frame=cv2.rectangle(frame,(0,40),(300,400),255,2)
            # frame=cv2.putText(frame,"Active Region",(75,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,255,2)
            image, results = mediapipe_detection(cropframe, hands)
            # print(results)
        
            # Draw landmarks
            # draw_styled_landmarks(image, results)
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try: 
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                
                
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    accuracy.append(str(res[np.argmax(res)]*100))
                            else:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)]*100)) 

                    if len(sentence) > 1: 
                        sentence = sentence[-1:]
                        accuracy=accuracy[-1:]

                    # Viz probabilities
                    # frame = prob_viz(res, actions, frame, colors,threshold)
            except Exception as e:
                # print(e)
                pass
            
            cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1)
            cv2.putText(frame,"Output: -"+' '.join(sentence)+''.join(accuracy), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
            # Show to screen
            cv2.imshow('OpenCV Feed', frame)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
@app.route('/video_live')
def video_live():
    return  Response(handgetsure(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/Wordreg")
def Wordreg():
    return render_template('tut3.html')

def wordregonation(file_path):
    pt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    img = cv2.imread(file_path)

    if img is not None:
        text = pt.image_to_string(img)
        return text
    else:
        return "Error: Could not read the image file."

# ...

@app.route("/wordre", methods=["POST"])
def wordre():
    word_result = None

    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return "error"
        file = request.files['file']
        # if the user does not select a file, the browser also
        # submits an empty part without a filename
        if file.filename == '':
            flash('No selected file')
            return "error"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            word_result = wordregonation(file_path)

    return render_template('tut3.html', word_result=word_result)

@app.route("/imagetextregT")
def imagetextregT():
    return render_template('tut4.html')

def imagetextregoniationT(file_path):
    pt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    path = os.path.abspath(file_path)

    image = Image.open(path)
    text = pt.image_to_string(image, lang='tam')

    print(text)

    # Save the text result to a file
    result_file_path = os.path.splitext(path)[0] + '_result.txt'
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        result_file.write(text)

    return result_file_path  # Returning the file path

@app.route("/tamil", methods=["POST"])
def tamil():
    word_result = None

    if request.method == "POST":
        if 'filet' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['filet']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            file.save(file_path)
            word_result = imagetextregoniationT(file_path)

    return render_template('tut4.html', word_result=word_result)

@app.route("/imagetextregH")
def imagetextregH():
    return render_template('tut5.html')

def imagetextregoniationH(file_path):
    pt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    path = os.path.abspath(file_path)

    image = Image.open(path)
    text = pt.image_to_string(image, lang='hin')

    print(text)

    # Save the text result to a file
    result_file_path = os.path.splitext(path)[0] + '_result.txt'
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        result_file.write(text)

    return result_file_path  # Returning the file path

@app.route("/hindi", methods=["POST"])
def hindi():
    word_result = None

    if request.method == "POST":
        if 'fileh' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['fileh']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            file.save(file_path)
            word_result = imagetextregoniationH(file_path)

    return render_template('tut5.html', word_result=word_result)

@app.route("/imagetextregE")
def imagetextregE():
    return render_template('tut6.html')

def imagetextregoniationE(file_path):
    pt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    path = os.path.abspath(file_path)

    image = Image.open(path)
    text = pt.image_to_string(image, lang='eng')

    print(text)

    # Save the text result to a file
    result_file_path = os.path.splitext(path)[0] + '_result.txt'
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        result_file.write(text)

    return result_file_path  # Returning the file path

@app.route("/english", methods=["POST"])
def english():
    word_result = None

    if request.method == "POST":
        if 'fileE' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['fileE']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            file.save(file_path)
            word_result = imagetextregoniationE(file_path)

    return render_template('tut6.html', word_result=word_result)

@app.route("/predictionumeric")
def predictionumeric():
    return render_template('tut8.html')
def numeric_value_recognition():
    global left_button_down, right_button_down, crop_preview, display, best_predictions

    def clear_whiteboard(display):
        wb_x1, wb_x2, wb_y1, wb_y2 = whiteboard_region["x"][0], whiteboard_region["x"][1], whiteboard_region["y"][0], whiteboard_region["y"][1] 
    
        display[wb_y1-10:wb_y2+12, wb_x1-10:wb_x2+12] = (255, 255, 255)

    def setup_display():
        title = np.zeros((80, 950, 3), dtype=np.uint8)
        board = np.zeros((600, 650, 3), dtype=np.uint8)
        panel = np.zeros((600, 300, 3), dtype=np.uint8)
        board[5:590, 8:645] = (255, 255, 255)
    
        board = cv2.rectangle(board, (8, 5), (645, 590), (127, 0, 225), 3)
        panel = cv2.rectangle(panel, (1, 4), (290, 590), (255, 215, 0), 2)
        panel = cv2.rectangle(panel, (22, 340), (268, 560), (255, 215, 0), 1)
        panel = cv2.rectangle(panel, (22, 65), (268, 280), (255, 215, 0), 1)
    
        cv2.line(panel, (145, 340), (145, 560), (255, 215, 0), 1)
        cv2.line(panel, (22, 380), (268, 380), (255, 215, 0), 1)
    
        cv2.putText(title, "       " +  window_name,(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(panel, "Action: ", (23, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(panel, "Best Predictions", (52, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(panel, "Prediction", (42, 362), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, "Accuracy", (168, 362), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, actions[0], (95, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_colors[actions[0]], 1)

        display = np.concatenate((board, panel), axis=1)
        display = np.concatenate((title, display), axis=0)
    
        return display

    def setup_panel(display):
        action_region_pt1, action_region_pt2 = status_regions["action"]
        preview_region_pt1, preview_region_pt2 = status_regions["preview"]
        label_region_pt1, label_region_pt2 = status_regions["labels"]
        acc_region_pt1, acc_region_pt2 = status_regions["accs"]
    
        display[action_region_pt1[1]:action_region_pt2[1], action_region_pt1[0]:action_region_pt2[0]] = (0, 0, 0)
        display[preview_region_pt1[1]:preview_region_pt2[1], preview_region_pt1[0]:preview_region_pt2[0]] = (0, 0, 0)
        display[label_region_pt1[1]:label_region_pt2[1], label_region_pt1[0]:label_region_pt2[0]] = (0, 0, 0)
        display[acc_region_pt1[1]:acc_region_pt2[1], acc_region_pt1[0]:acc_region_pt2[0]] = (0, 0, 0)
    
        if crop_preview is not None:
            display[preview_region_pt1[1]:preview_region_pt2[1], preview_region_pt1[0]:preview_region_pt2[0]] = cv2.resize(crop_preview, (crop_preview_h, crop_preview_w)) 
    
        if best_predictions:
            labels = list(best_predictions.keys())
            accs = list(best_predictions.values())
            prediction_status_cordinate = [
                ((725, 505), (830, 505), (0, 0, 255)),
                ((725, 562), (830, 562), (0, 255, 0)),
                ((725, 619), (830, 619), (255, 0, 0))
            ]
            for i in range(len(labels)):
                label_cordinate, acc_cordinate, color = prediction_status_cordinate[i]
            
                cv2.putText(display, labels[i], label_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, str(accs[i]), acc_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
            for i in range(len(labels), 3):
                label_cordinate, acc_cordinate, color = prediction_status_cordinate[i]
            
                cv2.putText(display, "_", label_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, "_", acc_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
        cv2.putText(display, current_action, (745, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_colors[current_action], 1)

    def arrange_crop_rectangle_cordinates(cor1, cor2):
        if cor1 is None or cor2 is None:
            return
    
        result = ()
        if cor1[1] < cor2[1]:
            if cor1[0] > cor2[0]:
                result = ( (cor2[0], cor1[1]), (cor1[0], cor2[1]) )
            else:
                result = (cor1, cor2)
        else:
            if cor2[0] > cor1[0]:
                result = ( (cor1[0], cor2[1]), (cor2[0], cor1[1]) )
            else:
                result = (cor2, cor1)
        return result

    def mouse_click_event(event, x, y, flags, params):
        if current_action == actions[1]:
            whiteboard_draw(event, x, y)
        elif current_action == actions[2]:
            character_crop(event, x, y)

    def whiteboard_draw(event, x, y):
        global left_button_down, right_button_down
    
        wb_x1, wb_x2, wb_y1, wb_y2 = whiteboard_region["x"][0], whiteboard_region["x"][1], whiteboard_region["y"][0], whiteboard_region["y"][1] 
    
        if event is cv2.EVENT_LBUTTONUP:
            left_button_down = False
        elif event is cv2.EVENT_RBUTTONUP:
            right_button_down = False
        elif wb_x1 <= x <= wb_x2 and wb_y1 <= y <= wb_y2:
            color = (0, 0, 0)
            if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP, cv2.EVENT_MOUSEMOVE]:
                if event is cv2.EVENT_LBUTTONDOWN:
                    color = (0, 0, 0)
                    left_button_down = True
                elif left_button_down and event is cv2.EVENT_MOUSEMOVE:
                    color = (0, 0, 0)
                elif event is cv2.EVENT_RBUTTONDOWN:
                    color = (255, 255, 255)
                    right_button_down = True
                elif right_button_down and event is cv2.EVENT_MOUSEMOVE:
                    color = (255, 255, 255)
                else:
                    return

                cv2.circle(display, (x, y), 10, color, -1)
                cv2.imshow(window_name, display)

    def character_crop(event, x, y):
        global bound_rect_cordinates, lbd_cordinate, lbu_cordinate, crop_preview, display, best_predictions
    
        wb_x1, wb_x2, wb_y1, wb_y2 = whiteboard_region["x"][0], whiteboard_region["x"][1], whiteboard_region["y"][0], whiteboard_region["y"][1] 
    
        if wb_x1 <= x <= wb_x2 and wb_y1 <= y <= wb_y2:
            if event is cv2.EVENT_LBUTTONDOWN:
                lbd_cordinate = (x, y)
            elif event is cv2.EVENT_LBUTTONUP:
                lbu_cordinate = (x, y)

            if lbd_cordinate is not None and lbu_cordinate is not None:
                bound_rect_cordinates = arrange_crop_rectangle_cordinates(lbd_cordinate, lbu_cordinate)
            elif lbd_cordinate is not None:
                if event is cv2.EVENT_MOUSEMOVE:
                    mouse_move_cordinate = (x, y)
                    mouse_move_rect_cordinates = arrange_crop_rectangle_cordinates(lbd_cordinate, mouse_move_cordinate)
                    top_cordinate, bottom_cordinate = mouse_move_rect_cordinates[0], mouse_move_rect_cordinates[1]
                
                    display_copy = display.copy()
                    cropped_region = display_copy[top_cordinate[1]:bottom_cordinate[1], top_cordinate[0]:bottom_cordinate[0]]
                    filled_rect = np.zeros((cropped_region.shape[:]))
                    filled_rect[:, :, :] = (0, 150, 0)
                    filled_rect = filled_rect.astype(np.uint8)
                    cropped_rect = cv2.addWeighted(cropped_region, 0.3, filled_rect, 0.5, 1.0)
                
                    if cropped_rect is not None:
                        display_copy[top_cordinate[1]:bottom_cordinate[1], top_cordinate[0]:bottom_cordinate[0]] = cropped_rect
                        cv2.imwrite("captured/filled.jpg", display_copy)
                        cv2.imshow(window_name, display_copy)

            if bound_rect_cordinates is not None:
                top_cordinate, bottom_cordinate = bound_rect_cordinates[0], bound_rect_cordinates[1]
                crop_preview = display[top_cordinate[1]:bottom_cordinate[1], top_cordinate[0]:bottom_cordinate[0]].copy()
                crop_preview = np.invert(crop_preview)
                best_predictions = predict(model, crop_preview)
                display_copy = display.copy()
                bound_rect_cordinates = lbd_cordinate = lbu_cordinate = None
                setup_panel(display)
                cv2.imshow(window_name, display)
        elif event is cv2.EVENT_LBUTTONUP:
            lbd_cordinate = lbu_cordinate = None
            cv2.imshow(window_name, display)        

    def load_model(path):
        model = Sequential()

        model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (5, 5), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.25))

        model.add(BatchNormalization())
        model.add(Flatten())

        model.add(Dense(256, activation="relu"))
        model.add(Dense(10, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.load_weights(path)
    
        return model

    def predict(model, image):
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = image / 255.0
        image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
        prediction = model.predict(image)
        best_predictions = dict()
    
        for i in range(3):
            max_i = np.argmax(prediction[0])
            acc = round(prediction[0][max_i], 1)
            if acc > 0:
                label = labels[max_i]
                best_predictions[label] = acc
                prediction[0][max_i] = 0
            else:
                break
            
        return best_predictions

    left_button_down = False
    right_button_down = False
    bound_rect_cordinates = lbd_cordinate = lbu_cordinate = None
    whiteboard_region = {"x": (20, 632), "y": (98, 656)}
    window_name = "Numeric Value Recognition"
    best_predictions = dict()
    crop_preview_h, crop_preview_w = 238, 206
    crop_preview = None
    actions = ["N/A", "DRAW", "CROP"]
    action_colors = {
        actions[0]: (0, 0, 255),
        actions[1]: (0, 255, 0),
        actions[2]: (0, 255, 192)
    }
    current_action = actions[0]
    status_regions = {
        "action": ((736, 97), (828, 131)),
        "preview": ((676, 150), (914, 356)),
        "labels": ((678, 468), (790, 632)),
        "accs": ((801, 468), (913, 632))
    }
    model = load_model("C:/Users/jaat5/Downloads/Website_New/models1/best_val_loss_model.h5")

    display = setup_display()
    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, mouse_click_event)
    pre_action = None

    while True:
        k = cv2.waitKey(1)
        if k == ord('d') or k == ord('c'):
            if k == ord('d'):
                current_action = actions[1]
            elif k == ord('c'):
                current_action = actions[2]
            if pre_action != current_action:
                setup_panel(display)
                cv2.imshow(window_name, display)
                pre_action = current_action
        elif k == ord('e'):
            clear_whiteboard(display)
            setup_panel(display)
            cv2.imshow(window_name, display)
        elif k == ord('z'):
            break

    # Release OpenCV windows when the loop is exited
    cv2.destroyAllWindows()
@app.route('/video_numeric')
def video_numeric():
    return Response(numeric_value_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/predictionalpha")
def predictionalpha():
    return render_template('tut7.html')
def predictionalphabat():
    global left_button_down, right_button_down, crop_preview, display, best_predictions

    def clear_whiteboard(display):
        wb_x1, wb_x2, wb_y1, wb_y2 = whiteboard_region["x"][0], whiteboard_region["x"][1], whiteboard_region["y"][0], whiteboard_region["y"][1] 
    
        display[wb_y1-10:wb_y2+12, wb_x1-10:wb_x2+12] = (255, 255, 255)

    def setup_display():
        title = np.zeros((80, 950, 3), dtype=np.uint8)
        board = np.zeros((600, 650, 3), dtype=np.uint8)
        panel = np.zeros((600, 300, 3), dtype=np.uint8)
        board[5:590, 8:645] = (255, 255, 255)
    
        board = cv2.rectangle(board, (8, 5), (645, 590), (255, 0, 0), 3)
        panel = cv2.rectangle(panel, (1, 4), (290, 590), (255, 215, 0), 2)
        panel = cv2.rectangle(panel, (22, 340), (268, 560), (255, 215, 0), 1)
        panel = cv2.rectangle(panel, (22, 65), (268, 280), (255, 215, 0), 1)
    
        cv2.line(panel, (145, 340), (145, 560), (255, 255, 255), 1)
        cv2.line(panel, (22, 380), (268, 380), (255, 255, 255), 1)
    
        cv2.putText(title, "       " +  window_name,(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(panel, "Action: ", (23, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(panel, "Best Predictions", (52, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(panel, "Prediction", (42, 362), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, "Accuracy", (168, 362), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, actions[0], (95, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_colors[actions[0]], 1)

        display = np.concatenate((board, panel), axis=1)
        display = np.concatenate((title, display), axis=0)
    
        return display
        
    def setup_panel(display):
        action_region_pt1, action_region_pt2 = status_regions["action"]
        preview_region_pt1, preview_region_pt2 = status_regions["preview"]
        label_region_pt1, label_region_pt2 = status_regions["labels"]
        acc_region_pt1, acc_region_pt2 = status_regions["accs"]
    
        display[action_region_pt1[1]:action_region_pt2[1], action_region_pt1[0]:action_region_pt2[0]] = (0, 0, 0)
        display[preview_region_pt1[1]:preview_region_pt2[1], preview_region_pt1[0]:preview_region_pt2[0]] = (0, 0, 0)
        display[label_region_pt1[1]:label_region_pt2[1], label_region_pt1[0]:label_region_pt2[0]] = (0, 0, 0)
        display[acc_region_pt1[1]:acc_region_pt2[1], acc_region_pt1[0]:acc_region_pt2[0]] = (0, 0, 0)
    
        if crop_preview is not None:
            display[preview_region_pt1[1]:preview_region_pt2[1], preview_region_pt1[0]:preview_region_pt2[0]] = cv2.resize(crop_preview, (crop_preview_h, crop_preview_w)) 
    
        if best_predictions:
            labels = list(best_predictions.keys())
            accs = list(best_predictions.values())
            prediction_status_cordinate = [
                ((725, 505), (830, 505), (0, 0, 255)),
                ((725, 562), (830, 562), (0, 255, 0)),
                ((725, 619), (830, 619), (255, 0, 0))
            ]
            for i in range(len(labels)):
                label_cordinate, acc_cordinate, color = prediction_status_cordinate[i]
            
                cv2.putText(display, labels[i], label_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, str(accs[i]), acc_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
            for i in range(len(labels), 3):
                label_cordinate, acc_cordinate, color = prediction_status_cordinate[i]
            
                cv2.putText(display, "_", label_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, "_", acc_cordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
        cv2.putText(display, current_action, (745, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_colors[current_action], 1)

    def arrange_crop_rectangle_cordinates(cor1, cor2):
        if cor1 is None or cor2 is None:
            return
    
        result = ()
        if cor1[1] < cor2[1]:
            if cor1[0] > cor2[0]:
                result = ( (cor2[0], cor1[1]), (cor1[0], cor2[1]) )
            else:
                result = (cor1, cor2)
        else:
            if cor2[0] > cor1[0]:
                result = ( (cor1[0], cor2[1]), (cor2[0], cor1[1]) )
            else:
                result = (cor2, cor1)
        return result

    def mouse_click_event(event, x, y, flags, params):
        if current_action == actions[1]:
            whiteboard_draw(event, x, y)
        elif current_action == actions[2]:
            character_crop(event, x, y)

    def whiteboard_draw(event, x, y):
        global left_button_down, right_button_down
    
        wb_x1, wb_x2, wb_y1, wb_y2 = whiteboard_region["x"][0], whiteboard_region["x"][1], whiteboard_region["y"][0], whiteboard_region["y"][1] 
    
        if event is cv2.EVENT_LBUTTONUP:
            left_button_down = False
        elif event is cv2.EVENT_RBUTTONUP:
            right_button_down = False
        elif wb_x1 <= x <= wb_x2 and wb_y1 <= y <= wb_y2:
            color = (0, 0, 0)
            if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP, cv2.EVENT_MOUSEMOVE]:
                if event is cv2.EVENT_LBUTTONDOWN:
                    color = (0, 0, 0)
                    left_button_down = True
                elif left_button_down and event is cv2.EVENT_MOUSEMOVE:
                    color = (0, 0, 0)
                elif event is cv2.EVENT_RBUTTONDOWN:
                    color = (255, 255, 255)
                    right_button_down = True
                elif right_button_down and event is cv2.EVENT_MOUSEMOVE:
                    color = (255, 255, 255)
                else:
                    return

                cv2.circle(display, (x, y), 10, color, -1)
                cv2.imshow(window_name, display)

    def character_crop(event, x, y):
        global bound_rect_cordinates, lbd_cordinate, lbu_cordinate, crop_preview, display, best_predictions

        wb_x1, wb_x2, wb_y1, wb_y2 = whiteboard_region["x"][0], whiteboard_region["x"][1], whiteboard_region["y"][0], whiteboard_region["y"][1] 
    
        if wb_x1 <= x <= wb_x2 and wb_y1 <= y <= wb_y2:
            if event is cv2.EVENT_LBUTTONDOWN:
                lbd_cordinate = (x, y)
            elif event is cv2.EVENT_LBUTTONUP:
                lbu_cordinate = (x, y)

            if lbd_cordinate is not None and lbu_cordinate is not None:
                bound_rect_cordinates = arrange_crop_rectangle_cordinates(lbd_cordinate, lbu_cordinate)
            elif lbd_cordinate is not None:
                if event is cv2.EVENT_MOUSEMOVE:
                    mouse_move_cordinate = (x, y)
                    mouse_move_rect_cordinates = arrange_crop_rectangle_cordinates(lbd_cordinate, mouse_move_cordinate)
                    top_cordinate, bottom_cordinate = mouse_move_rect_cordinates[0], mouse_move_rect_cordinates[1]
                
                    display_copy = display.copy()
                    cropped_region = display_copy[top_cordinate[1]:bottom_cordinate[1], top_cordinate[0]:bottom_cordinate[0]]
                    filled_rect = np.zeros((cropped_region.shape[:]))
                    filled_rect[:, :, :] = (0, 255, 0)
                    filled_rect = filled_rect.astype(np.uint8)
                    cropped_rect = cv2.addWeighted(cropped_region, 0.3, filled_rect, 0.5, 1.0)
                
                    if cropped_rect is not None:
                        display_copy[top_cordinate[1]:bottom_cordinate[1], top_cordinate[0]:bottom_cordinate[0]] = cropped_rect
                        cv2.imwrite("captured/filled.jpg", display_copy)
                        cv2.imshow(window_name, display_copy)

            if bound_rect_cordinates is not None:
                top_cordinate, bottom_cordinate = bound_rect_cordinates[0], bound_rect_cordinates[1]
                crop_preview = display[top_cordinate[1]:bottom_cordinate[1], top_cordinate[0]:bottom_cordinate[0]].copy()
                crop_preview = np.invert(crop_preview)
                best_predictions = predict(model, crop_preview)
                display_copy = display.copy()
                bound_rect_cordinates = lbd_cordinate = lbu_cordinate = None
                setup_panel(display_copy)
                cv2.imshow(window_name, display_copy)
        elif event is cv2.EVENT_LBUTTONUP:
            lbd_cordinate = lbu_cordinate = None
            cv2.imshow(window_name, display)        

    def load_model(path):
        model = Sequential()

        model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (5, 5), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.25))

        model.add(BatchNormalization())
        model.add(Flatten())

        model.add(Dense(256, activation="relu"))
        model.add(Dense(26, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.load_weights(path)
    
        return model

    def predict(model, image):
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = image / 255.0
        image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
        prediction = model.predict(image)
        best_predictions = dict()
    
        for i in range(3):
            max_i = np.argmax(prediction[0])
            acc = round(prediction[0][max_i], 1)
            if acc > 0:
                label = labels[max_i]
                best_predictions[label] = acc
                prediction[0][max_i] = 0
            else:
                break
            
        return best_predictions

    left_button_down = False
    right_button_down = False
    bound_rect_cordinates = lbd_cordinate = lbu_cordinate = None
    whiteboard_region = {"x": (20, 632), "y": (98, 656)}
    window_name = "Character Value Recognition"
    best_predictions = dict()
    crop_preview_h, crop_preview_w = 238, 206
    crop_preview = None
    actions = ["N/A", "DRAW", "CROP"]
    action_colors = {
        actions[0]: (0, 0, 255),
        actions[1]: (0, 255, 0),
        actions[2]: (0, 255, 192)
    }
    current_action = actions[0]
    status_regions = {
        "action": ((736, 97), (828, 131)),
        "preview": ((676, 150), (914, 356)),
        "labels": ((678, 468), (790, 632)),
        "accs": ((801, 468), (913, 632))
    }
    model = load_model("C:/Users/jaat5/Downloads/Website_New/models/best_val_loss_model.h5")

    display = setup_display()
    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, mouse_click_event)
    pre_action = None

    while True:
        k = cv2.waitKey(1)
        if k == ord('d') or k == ord('c'):
            if k == ord('d'):
                current_action = actions[1]
            elif k == ord('c'):
                current_action = actions[2]
            if pre_action != current_action:
                setup_panel(display)
                cv2.imshow(window_name, display)
                pre_action = current_action
        elif k == ord('e'):
            clear_whiteboard(display)
            setup_panel(display)
            cv2.imshow(window_name, display)
        elif k == ord('z'):
            break

    # Release OpenCV windows when the loop is exited
    cv2.destroyAllWindows()

@app.route('/video_moving')
def video_moving():
    return Response(predictionalphabat(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        new_user = User(username=form.username.data, password=form.password.data)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data, password=form.password.data).first()

        if user:
            # Redirect to the desired page after successful login
            return redirect(url_for('home'))
        else:
            return 'Invalid login credentials'

    return render_template('login.html', form=form)

@app.route('/users')
def display_users():
    users = User.query.all()
    return render_template('users.html', users=users)

@app.route('/submit_form', methods=['GET', 'POST'])
def submit_form():
    form = ContactForm()

    if form.validate_on_submit():
        # If the form is valid, create a new Contact instance and add it to the database
        new_contact = Contact(name=form.name.data, email=form.email.data, message=form.message.data)
        db.session.add(new_contact)
        db.session.commit()

        return redirect(url_for('home'))

    return render_template('form.html', form=form)

@app.route('/success')
def success():
    return "Form submitted successfully!"

@app.route('/view_data')
def view_data():
    contacts = Contact.query.all()
    return render_template('view_data.html', contacts=contacts)

def load_captcha_model():
    return load_model('captcha_model.h5')

def predict(model, filepath):
    symbols = string.ascii_lowercase + "0123456789" # All symbols captcha can contain
    num_symbols = len(symbols)
    img_shape = (50, 200, 1)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0
    else:
        print("Not detected")
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (5, 36))
    l_ind = []
    probs = []
    for a in ans:
        l_ind.append(np.argmax(a))
        # probs.append(np.max(a))

    capt = ''
    for l in l_ind:
        capt += symbols[l]
    return capt

@app.route("/captcha")
def captcha():
    return render_template('tut9.html')

@app.route("/captch", methods=["GET", "POST"])
def captch():
    word_result = None

    if request.method == "POST":
        if 'filec' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['filec']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            file.save(file_path)
            
            # Load the model and predict
            model = load_captcha_model()
            word_result = predict(model, file_path)

    return render_template('tut9.html', word_result=word_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 

                     
            
