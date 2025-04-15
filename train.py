from utils.conllu_loader import load_conllu_data
from model.pos_lstm import build_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load dataset
train_sents, train_tags = load_conllu_data("hi_hdtb-ud-train.conllu")
test_sents, test_tags = load_conllu_data("hi_hdtb-ud-test.conllu")

# Tokenizers for words and tags
word_tokenizer = Tokenizer(oov_token="<OOV>")
word_tokenizer.fit_on_texts(train_sents)
tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(train_tags)

# Convert to sequences
X_train = word_tokenizer.texts_to_sequences(train_sents)
y_train = tag_tokenizer.texts_to_sequences(train_tags)

# Pad sequences
max_len = max(len(seq) for seq in X_train)
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
y_train = pad_sequences(y_train, maxlen=max_len, padding='post')

# One-hot encode y labels
y_train = [to_categorical(i, num_classes=len(tag_tokenizer.word_index)+1) for i in y_train]
y_train = np.array(y_train)

# Build and train the model
model = build_model(len(word_tokenizer.word_index)+1, len(tag_tokenizer.word_index)+1, max_len)
model.summary()
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)

# Save the model and tokenizers
model.save("pos_lstm_model.h5")
import pickle
with open("word_tokenizer.pkl", "wb") as f:
    pickle.dump(word_tokenizer, f)
with open("tag_tokenizer.pkl", "wb") as f:
    pickle.dump(tag_tokenizer, f)
