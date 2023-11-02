import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
#nltk.download('stopwords')
import nltk as nnn
ps = PorterStemmer()
# Data preprocessing Function
def transf_text(text):
    text =  text.lower()
    text = nnn.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open("/content/vectorizer.pkl",'rb'))
model = pickle.load(open("/content/model.pkl",'rb'))

st.title("Email/Sms Spam Classifier")

input_sms = st.text_area("Enter the Message")
if st.button('Predict'):
    transformed_sms = transf_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
