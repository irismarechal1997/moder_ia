import streamlit as st
from PIL import Image
from IPython.display import display
import requests
from api.fast import predict_binary, generate_fight_tweet, predict_classif
import base64
import os
import openai
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components


#### SET CSS and main functions
st.set_page_config(
    page_title="Let's fight online Hate-Speech",
    page_icon="✊",
    layout="wide",
)

# Define color palette
colors = {
    "blue": "#3498db",
    "green": "#2ecc71",
    "orange": "#e67e22",
    "purple": "#9b59b6",
    "red": "#e74c3c",
}

st.markdown('<style>body {background-image: url("Background_site.png"); background-size: cover; background-repeat: no-repeat; background-attachment: fixed;}</style>', unsafe_allow_html=True)


# Set a background image
image_path = "Background_site.png"

# Create a CSS rule to set the background image
background_css = f"""
<style>
body {{
    background-image: url('{image_path}');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
"""

# Inject the CSS rule into the page
st.markdown(background_css, unsafe_allow_html=True)

# Modification des paramètres visuels du site
primaryColor = "#F63366"
textColor = "#262730"
font = "sans serif"


# Function to encode a file to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#########################################

# Part 1 : Presentation of the project
st.title("✊ Let's Fight Online Hate-Speech")
st.markdown(
    "We are 3 students from Le Wagon learning data science, machine learning and deep learning.As part of our final project, we decided to create a small program allowing anyone to enter a tweet, get infromation on the level of offensiveness of the tweet, and generate automatically an appropriate tweet response.")

# Replace 'your_presentation.pdf' with the actual file path of your presentation
presentation_file = 'your_presentation.pdf' # add presentation file
presentation_button_label = "Download Project Presentation"

if st.button(presentation_button_label):
    presentation_base64 = get_base64_of_bin_file(presentation_file)
    st.markdown(f'<a href="data:application/pdf;base64,{presentation_base64}" download="{presentation_file}">Click here to download the presentation</a>', unsafe_allow_html=True)

#########################################


# Part 2: Tweet Analysis
st.header("1️⃣ Analyze a Tweet")
tweet = st.text_area("Write down your tweet")

if tweet:
    st.write("You entered:")
    st.write(tweet)

    # Add a button to check the tweet
if st.button("Check Tweet"):
    if tweet:
        # Replace this with your API call to check for offensiveness
        response = predict_binary(tweet)  # Replace with your API call
        st.write("API Response:", response["type of tweet"])

        # If the API response is "the tweet is offensive," show the "Learn more" button
        if response == "the tweet is offensive":
            if st.button("Learn more on my tweet"):
                # Scroll to Part 3
                st.write(
                    "<a href='#tweet-description'>Scroll down to learn more about the tweet.</a>",
                    unsafe_allow_html=True,
                )
    else:
        st.write("Please enter a tweet to check.")


############################################

# Part 3: Tweet Description
st.header("2️⃣ Tweet Description")

if st.button("Analyze Tweet"):

    # Add code here to call your second API to describe the tweet
    response_class = predict_classif(tweet)
    description_result = f"This tweet has been analyzed as a {response_class['label']}"  # Replace with actual result
    st.write(description_result)

()
############################################
# Part 4: Generate Response to Tweet
st.header("3️⃣ Generate Response to Tweet")
if st.button("Generate Response to Tweet"):
    # Add code here to call your third API to generate a response
    response_class = predict_classif(tweet)

    openai.api_key = os.environ.get("API_KEY")
    content_of_the_request = f"We have received an offensive tweet. This tweet can be classified as {response_class['label']}. Please find here the tweet '{tweet}'. Could you please generate a response to this tweet by explaining that this tweet is {response_class['label']} and recall the potential penalties incurred (legally but also in terms of banning on the tweeter platform). Please generate a response in the form of a tweet of max 280 characters and directly generate the quoted response without anything else."
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role':'user','content': content_of_the_request}])
    st.write(response.choices[0].message.content)
    #response = generate_fight_tweet(tweet, response_class['label'])
    #st.write(generate_fight_tweet(tweet, response_class['label']))
    #response_result = response  # Replace with actual result
    #st.write(response_result)


###############################################
#Part 5 : Who we are ?
st.title("Who are we ? ")
st.markdown("Meet the team behind the project:")


# Create a layout for each team member
def team_member(image_path, name, description):
    st.image(image_path, use_column_width='auto', caption=name)
    st.markdown(f'<p style="text-align: center;">{name}</p>', unsafe_allow_html=True)

# Create a row with three columns
col1, col2, col3 = st.columns(3)

# Team Member 1
with col1:
    team_member("IMG_0606.png", "Marianne", "Team Member 1 - Description")

# Team Member 2
with col2:
    team_member("IMG_0607.png", "Team Member 2", "Team Member 2 - Description")

# Team Member 3
with col3:
    team_member("PP_Iris.png", "Team Member 3", "Team Member 3 - Description")
