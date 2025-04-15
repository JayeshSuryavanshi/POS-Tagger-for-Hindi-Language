from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, TimeDistributed, Dense

def build_model(vocab_size, tag_size, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len))
    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed(Dense(tag_size, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
