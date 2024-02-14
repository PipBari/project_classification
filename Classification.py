import atexit
import os
import shutil

from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

from ImageClassifier import ImageClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

classifier = ImageClassifier()


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    image_url = None
    prediction_text = "Файл не загружен"
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image = Image.open(file_path)
            predicted_class, confidence = classifier.classify_image(image)
            prediction_text = f'Результат: {predicted_class} с уверенностью {confidence:.2f}%'
            image_url = url_for('static', filename=os.path.join('uploads', filename).replace('\\', '/'))

    return render_template('index.html', prediction=prediction_text, image_url=image_url)


def cleanup_directory():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


atexit.register(cleanup_directory)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
