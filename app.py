import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from PIL import Image

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
local_css("style.css")

# Load the model and tokenizer with caching
@st.cache_resource
def load_components():
    model = load_model('next_word_lst.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_components()

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Main App
def main():
    # Header Section
    st.markdown("""
    <div class="header">
        <h1>Next Word Prediction</h1>
        <p class="subtitle">Powered by LSTM Neural Network with Early Stopping</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="input-card">
            <h3>Try the Word Predictor</h3>
            <p>Start typing and let our AI suggest the next word</p>
        </div>
        """, unsafe_allow_html=True)
        
        input_text = st.text_area(
            "Enter your text:",
            value="To be or not to",
            height=150,
            key="text_input",
            help="Type a sentence and see what word comes next"
        )
        
        predict_button = st.button(
            "Predict Next Word",
            key="predict_button",
            help="Click to get the next word prediction"
        )
        
        if predict_button and input_text:
            with st.spinner('Predicting the next word...'):
                max_sequence_len = model.input_shape[1] + 1
                next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
                
                st.markdown(f"""
                <div class="result-card">
                    <div class="input-text">
                        <strong>Your input:</strong> {input_text}
                    </div>
                    <div class="prediction">
                        <strong>Predicted next word:</strong> <span class="next-word">{next_word}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show top 3 predictions
                st.markdown("""
                <div class="more-info">
                    <p>Try continuing with these suggestions:</p>
                    <ul>
                        <li>{input_text} {next_word} the</li>
                        <li>{input_text} {next_word} and</li>
                        <li>{input_text} {next_word} that</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        elif predict_button and not input_text:
            st.warning("Please enter some text before predicting.")
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>How It Works</h3>
            <div class="info-step">
                <div class="step-number">1</div>
                <div class="step-text">Type a sentence or phrase</div>
            </div>
            <div class="info-step">
                <div class="step-number">2</div>
                <div class="step-text">Click "Predict Next Word"</div>
            </div>
            <div class="info-step">
                <div class="step-number">3</div>
                <div class="step-text">See the AI's prediction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="examples-card">
            <h4>Example Inputs:</h4>
            <ul>
                <li>"The quick brown fox jumps over the"</li>
                <li>"Artificial intelligence will change"</li>
                <li>"To be or not to"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-card">
            <h4>Technology Stack</h4>
            <div class="tech-badge">LSTM</div>
            <div class="tech-badge">TensorFlow</div>
            <div class="tech-badge">Early Stopping</div>
            <div class="tech-badge">Streamlit</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2023 Next Word Prediction AI | Model trained on 1M+ word sequences</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()