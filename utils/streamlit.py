import streamlit as st
import numpy as np
import pandas as pd

# Set the background color to light blue

primaryColor="#F63366"
backgroundColor="#F8F0E5"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"


#st.markdown(
    """
    <style>
    body {
        background-color: #f0f5f5;
    }
    </style>
    """,
    #unsafe_allow_html=True
#)

# Add a title to the page
st.title("Fighting against online hate speech")

# Add a text input field for entering a tweet
tweet = st.text_area("Write down your tweet")

# Display the entered tweet
if tweet:
    st.write("You entered:")
    st.write(tweet)
