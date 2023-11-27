import streamlit as st
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences



st.title("DL_PREDICTION")

# Dropdown for selecting the task
task = st.selectbox("Select Task", ["Sentiment Analysis", "Tumor Classification"])

if task == "Sentiment Analysis":
    selected_algorithm = st.selectbox("Select Algorithm", ["Perceptron", "Backpropagation", "DNN", "RNN", "LSTM"])
    if selected_algorithm=='LSTM':
        imdb_lstm=load_model('saved_model/imdb_lstm.h5')
    
        def lstm_predict_imdb(review, model, max_review_length):
            top_words=5000
            # Convert the review to the IMDB dataset format
            review_sequence = imdb.get_word_index()
            review = [review_sequence[word] if word in review_sequence and review_sequence[word] < top_words else 0 for word in review.split()]
            review = sequence.pad_sequences([review], maxlen=max_review_length)
            prediction = model.predict(review)
            if prediction>0.5:
                st.write("Positive Sentiment")
            else:
                st.write("Negative Sentiment")
        review=st.text_input("Enter your review here.")
        if st.button("*Make Prediction*"):
            lstm_predict_imdb(review,imdb_lstm,500)
    elif selected_algorithm=="Perceptron":
        imdb_perceptron = joblib.load(r"saved_model/imdb_percepton.joblib")

        def perceptron_predict_imdb(review, vectorizer, model):
            review_bow = vectorizer.transform([review])
            prediction = imdb_perceptron.predict(review_bow)
            if prediction > 0.5 :
                st.success("Positive sentiment")
            else:
                st.success("Negative sentiment")
        
        review=st.text_input("Give your review here.")
        vectorizer = joblib.load(r"saved_model/vectariser_imdb.joblib")
        if st.button("*Make Prediction*"):
            perceptron_predict_imdb(review,vectorizer,imdb_perceptron)
    elif selected_algorithm=='DNN':
        imdb_dnn = load_model(r'saved_model/imdb_dnn.h5') 
        word_to_index = imdb.get_word_index()
        def dnn_predict_sentiment(review, model):
            prediction = imdb_dnn.predict(review)
            if prediction > 0.5 :
                st.success("Positive sentiment")
            else:
                st.success("Negative sentiment")
        
        review=st.text_input("Enter your review here.")
        
        if st.button("Make Prediction"):
            max_review_length=500
            new_review_tokens=[word_to_index.get(word, 0) for word in review.split()]
            new_review_tokens = pad_sequences([new_review_tokens], maxlen=max_review_length)    
            dnn_predict_sentiment(new_review_tokens,imdb_dnn)
    elif selected_algorithm=='RNN':
    
        imdb_rnn=load_model("saved_model/rnn_imdb.h5")

        def rnn_predict_imdb(review, model, max_review_length):
            top_words=5000
            # Convert the review to the IMDB dataset format
            review_sequence = imdb.get_word_index()
            review = [review_sequence[word] if word in review_sequence and review_sequence[word] < top_words else 0 for word in review.split()]
            review = sequence.pad_sequences([review], maxlen=max_review_length)
            prediction = model.predict(review)
            if prediction>0.5:
                st.write("Positive Sentiment")
            else:
                st.write("Negative Sentiment")

        review=st.text_input("Enter your review here.")
        if st.button("*Make Prediction*"):
            rnn_predict_imdb(review,imdb_rnn,500)
elif task == "Tumor Classification":
    selected_algorithm = st.selectbox("Select Algorithm", ["CNN"])
    uploaded_file = st.file_uploader("Upload Tumor Image", type=["jpg", "jpeg", "png"])

    tumor_cnn = load_model('saved_model/cnn_tumor.h5')

    def make_tumor_prediction(img,model):
        if img is None:
            st.warning("Please upload an image.")
            return
        if img is not None:
            img_array = np.array(Image.open(img))
            img_array = cv2.resize(img_array, (128, 128))
            input_img = np.expand_dims(img_array, axis=0)
            prediction = model.predict(input_img)
            if prediction > 0.5:
                st.success("Tumor Detected")
            else:
                st.success("No Tumor")

    if st.button('Make prediction'):
        make_tumor_prediction(uploaded_file, tumor_cnn)
