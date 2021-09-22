import streamlit as st
st.title("Hello")
st.subheader("welcome")


import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout,Dense,Flatten,Embedding,LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from gtts import gTTS
from IPython.display import Audio
 
def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]',' ',sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+",' ',sentence)
    sentence = re.sub(r'\s+',' ',sentence)
    return sentence
 
TAG_RE = re.compile(r'<[^>]+>')
 
def remove_tags(text):
    return TAG_RE.sub('',text)
 
import pickle

def predict(tokenizer,model,comment):
    text = comment
    tts = gTTS("you are actually mad")
    tts.save("check.wav")
    audio = "check.wav"
    #Audio(audio)
    st.audio(audio)
    text = preprocess_text(text)
    text = [text.split(" ")]
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text)
    prediction=model.predict(text)
    value = prediction[0][0]
    if value<0.3:
        return "Negative"
    elif value<0.7:
        return "Neutral"
    else:
        return "Positive"
    
def main():
    with open("model.pkl", 'rb') as handle:
        model = pickle.load(handle)

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    comment = st.text_input("Enter your comment","Type here...")
    prediction=""
    if st.button("Submit"):
        prediction = predict(tokenizer,model,comment)
    st.success("Sentiment of your comment is : "+prediction)
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
