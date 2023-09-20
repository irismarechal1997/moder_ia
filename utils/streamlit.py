import streamlit as st
import numpy as np
import pandas as pd
import requests

# modification des paramètres visuels du site
primaryColor="#F63366"
textColor="#262730"
font="sans serif"

# ajout d'une photo dans le fond du site
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("data/Background_site.png");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Page 1: Fighting against online hate speech
def page_one():
    set_background()
    st.title("Fighting against online hate speech")

    # Add a text input field for entering a tweet
    tweet = st.text_area("Write down your tweet")
    # Display the entered tweet
    if tweet:
        st.write("You entered:")
        st.write(tweet)

    # Add a button to check the tweet
    if st.button("Check Tweet"):
        #if tweet:
            # Replace this with your API call to check for offensiveness
            #response = check_tweet_offensiveness(tweet)
            #st.write("API Response:", response)

            # If the API response is "the tweet is offensive," show the "Learn more" button
            #if response == "the tweet is offensive":
                st.button("Learn more on my tweet")


# Page 2: Learn more on my tweet
def page_two():
    set_background()
    st.title("Learn more on my tweet")
