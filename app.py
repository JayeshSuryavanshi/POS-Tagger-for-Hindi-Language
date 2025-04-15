from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizers
model = load_model("pos_lstm_model.h5")
with open("word_tokenizer.pkl", "rb") as f:
    word_tokenizer = pickle.load(f)
with open("tag_tokenizer.pkl", "rb") as f:
    tag_tokenizer = pickle.load(f)

index_to_tag = {v: k for k, v in tag_tokenizer.word_index.items()}
max_len = model.input_shape[1]

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    tagged = []
    sentence = ""
    if request.method == "POST":
        sentence = request.form["sentence"]
        tokens = sentence.strip().split()

        seq = word_tokenizer.texts_to_sequences([tokens])
        padded_seq = pad_sequences(seq, maxlen=max_len, padding="post")
        prediction = model.predict(padded_seq)
        pred_tags = np.argmax(prediction[0], axis=-1)

        for i, word in enumerate(tokens):
            tag = index_to_tag.get(pred_tags[i], 'O')
            tagged.append((word, tag))
    return render_template("index.html", tagged=tagged, sentence=sentence)

if __name__ == "__main__":
    app.run(debug=True)
