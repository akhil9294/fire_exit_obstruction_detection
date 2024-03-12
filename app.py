import os
from flask import Flask,request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from utils import f_resize_raw_image, f_predict, f_clear_folder, f_remove_files_in_directory
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import scipy

app=Flask(__name__)
CORS(app)
cors = CORS(app, resource={r"/*": {"origins": "*"}})


UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


TMP_FOLDER = "tmp"
if not os.path.exists(TMP_FOLDER):
    os.makedirs(TMP_FOLDER)

TMP_FOLDER0 = os.path.join(TMP_FOLDER, '0')
if not os.path.exists(TMP_FOLDER0):
    os.makedirs(TMP_FOLDER0)

OUTPUT_FOLDER = "results"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TMP_FOLDER"] = TMP_FOLDER

BASE_URL = os.environ.get("BASE_URL", "http://localhost:5003")
PORT = os.environ.get("PORT", 5003)

target_size = (224,224)


@app.route("/api/image/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)



@app.route("/obstruction_detection/process-image", methods = ['POST'])
def upload_file():
    if "file" not in request.files:
        return "No file part"
    
    image_file = request.files["file"]
    if image_file.filename == "":
        return "No selected image file"

    filename = secure_filename(image_file.filename)
    if image_file:
        image_file_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_file_path)
        
        
    
    if not (image_file_path.endswith(".jpg") or image_file_path.endswith(".jpeg") or image_file_path.endswith(".png")):
        return (
            jsonify(
                {
                    "message": "Incorrect file format of image file. Allowed only (jpg and png)",
                    "success": False,
                }
            ),
            200,
        )
    
    # Picking the image from ./uploads directory, resizing the image to target_size and pushing the resized images to tmp directory.
    
    f_resize_raw_image(UPLOAD_FOLDER, TMP_FOLDER0, target_size)

    # Clearing OUTPUT_FOLDER and Sending latest resized image to OUTPUT_FOLDER.
    f_remove_files_in_directory(OUTPUT_FOLDER)
    f_resize_raw_image(TMP_FOLDER0, OUTPUT_FOLDER, target_size)
    os.rename(os.path.join(OUTPUT_FOLDER,'0.jpg'), os.path.join(OUTPUT_FOLDER, filename))

    # Creating data generator
    image_datagen = ImageDataGenerator( rescale = 1.0/255. )
    image_generator = image_datagen.flow_from_directory(TMP_FOLDER, batch_size=10, class_mode='binary', target_size=target_size)
    prediction  = f_predict(image_generator)
    f_clear_folder(TMP_FOLDER0)
    f_clear_folder(UPLOAD_FOLDER)



    return (
            jsonify(
                {
                    "message": f"Image processed sucessfully",
                    "success": True,
                    "Obstruction_dectected": prediction,
                    "image_url": f"{BASE_URL}/api/image/{filename}",
                }
            ),
            200,
        )


if __name__ =="__main__":
    app.run(debug=True, port = PORT, host="0.0.0.0")