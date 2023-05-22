from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
model = tf.lite.Interpreter(model_path="./goat_model.tflite") # ganti "model.tflite" dengan path model anda
model.allocate_tensors()

class_labels = ['Healthy Goat', 'Pinkeye Goat']


def interpret_output(output):
    # Here, the output of the model is assumed to be a 2D array where the first
    # element of the first array is the probability of the 'Healthy Goat' class.
    # Adjust this if your model's output is structured differently.
    
    if output[0][0] > 0.5:
        prediction = 'Healthy Goat'
    else:
        prediction = 'Pinkeye Goat'

    return prediction


def preprocess_image(image_path):
    # Membuka dan memproses gambar ke bentuk yang dapat diterima oleh model
    img = Image.open(image_path)
    img = img.resize((150, 150))  # Ubah ini ke ukuran yang diharapkan model Anda
    img = np.array(img)
    img = img / 255.0  # Normalisasi nilai pixel ke rentang [0,1]
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch

    return img


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        imagefile = request.files['image']
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        # preprocess gambar sesuai kebutuhan model Anda di sini
        image = preprocess_image(image_path)
        image = image.astype('float32')

        # masukkan gambar ke model
        input_details = model.get_input_details()
        model.set_tensor(input_details[0]['index'], image)

        model.invoke()

        output_details = model.get_output_details()
        output = model.get_tensor(output_details[0]['index'])

        # interpretasi output sesuai kebutuhan model Anda
        prediction = interpret_output(output)

        return render_template('./index.html', prediction=prediction)
    return render_template('./index.html', prediction='')

if __name__ == '__main__':
    app.run(debug=True)
