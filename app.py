import streamlit as st
import numpy as np
import tensorflow as tf
import os
import urllib
import librosa

def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('Emotion Recognition', 'View Source Code')
    )

    if selected_box == 'Emotion Recognition':
        st.sidebar.success('To try it yourself, upload an audio file.')
        application()
    if selected_box == 'View Source Code':
        st.code(get_file_content_as_string("app.py"))

@st.cache_data(show_spinner=False)
def get_file_content_as_string(path):
    url = '' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

@st.cache_data(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model('mymodel.h5')
    
    # Create a new instance of the Adam optimizer
    custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile the model with the new optimizer
    model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def application():
    models_load_state = st.text('\nLoading models...')
    model = load_model()
    models_load_state.text('\nModels Loading...complete')
    
    file_to_be_uploaded = st.file_uploader("Choose an audio...", type="wav")
    
    if file_to_be_uploaded:
        st.audio(file_to_be_uploaded, format='audio/wav')
        emotion = predict(model, file_to_be_uploaded)
        st.success('Emotion of the audio is ' + emotion)

def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def predict(model, wav_filepath):
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    test_point = extract_mfcc(wav_filepath)
    test_point = np.reshape(test_point, newshape=(1, 40, 1))
    predictions = model.predict(test_point)
    emotion_label = emotions[np.argmax(predictions[0]) + 1]
    return emotion_label

if __name__ == "__main__":
    main()
