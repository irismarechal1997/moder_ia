import streamlit as st
from PIL import Image
from IPython.display import display
import requests
from api.fast import predict_binary, generate_fight_tweet
import base64


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

# Modification des paramètres visuels du site
primaryColor = "#F63366"
textColor = "#262730"
font = "sans serif"
image_path = "../data/Background_site.png"

# Function to encode a file to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# CSS for circular frames
circle_css = """
    display: flex;
    justify-content: center;
    align-items: center;
    width: 150px;
    height: 150px;
    background-color: white;
    border-radius: 50%;
    padding: 10px;
"""

#########################################

# Part 1 : Presentation of the project
st.title("Let's Fight Online Hate-Speech")
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
st.header("Analyze a Tweet")
tweet = st.text_area("Write down your tweet")

if tweet:
    st.write("You entered:")
    st.write(tweet)

    # Add a button to check the tweet
if st.button("Check Tweet"):
    if tweet:
        # Replace this with your API call to check for offensiveness
        response = predict_binary(tweet) #nom de la requête API pour le premier modèle
        st.write("API Response:", response)

        # If the API response is "the tweet is offensive," show the "Learn more" button
        if response == "the tweet is offensive":
            st.button("Learn more on my tweet")
            st.write("<a href='#tweet-description'>Scroll down to learn more about the tweet.</a>",unsafe_allow_html=True)
    return response


############################################

# Part 3: Tweet Description
st.header("Tweet Description")

if st.button("Analyze Tweet"):

    # Add code here to call your second API to describe the tweet

    description_result = "This tweet has been analyzed as 'racist'"  # Replace with actual result
    st.write(description_result)


############################################
# Part 4: Generate Response to Tweet
st.header("Generate Response to Tweet")
if st.button("Generate Response to Tweet"):
    # Add code here to call your third API to generate a response
    response = generate_fight_tweet(tweet, 'racist')
    response_result = response  # Replace with actual result
    st.write(response_result)


###############################################
#Part 5 : Who we are ?
st.title("Who are we ? ")
st.markdown("Meet the team behind the project:")

# Add pictures and descriptions of team members
with st.beta_container():
    st.image("image1.jpg", caption="Team Member 1", use_column_width=True, output_format="auto")
    st.write("Team Member 1 - Description")
    st.markdown(f'<div style="{circle_css}"><img src="image1.jpg" width="100" height="100"></div>', unsafe_allow_html=True)

with st.beta_container():
    st.image("image2.jpg", caption="Team Member 2", use_column_width=True, output_format="auto")
    st.write("Team Member 2 - Description")
    st.markdown(f'<div style="{circle_css}"><img src="image2.jpg" width="100" height="100"></div>', unsafe_allow_html=True)

with st.beta_container():
    st.image("image3.jpg", caption="Team Member 3", use_column_width=True, output_format="auto")
    st.write("Team Member 3 - Description")
    st.markdown(f'<div style="{circle_css}"><img src="image3.jpg" width="100" height="100"></div>', unsafe_allow_html=True)

# Main function to run the Streamlit app
#def main():
    #st.sidebar.title("Navigation")
    #page = st.sidebar.selectbox("Go to:", ["Home", "Learn More"])

    #if page == "Home":
        #page_one()
    #elif page == "Learn More":
        #page_two()

#if __name__ == "__main__":
    #main()
