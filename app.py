import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('svc_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_text = request.form['text']
        input_text_sp = [x.strip() for x in input_text.split(',')]
        np_data = np.asarray(input_text_sp, dtype=np.float32)

        if np_data.shape[0] != model.n_features_in_:
            return render_template("index.html", message=f"Expected {model.n_features_in_} inputs, got {np_data.shape[0]}.")

        prediction = model.predict(np_data.reshape(1, -1))

        if prediction == 1:
            output = "This person has Parkinson's disease"
        else:
            output = "This person does not have Parkinson's disease"

        return render_template("index.html", message=output)
    
    except Exception as e:
        return render_template("index.html", message=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
