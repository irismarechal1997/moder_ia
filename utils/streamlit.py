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

# Display the image at the top

st.set_page_config(
    page_title="Let's fight online Hate-Speech",
    page_icon="✊",
    layout="wide",
)

st.image('utils/bandeau.png', use_column_width=True)



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


# Function to encode a file to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#########################################

# Part 1 : Presentation of the project
st.markdown(
    "Hello everyone! We are three students from Le Wagon's Data Science program, where we have spent the past two months diving into the fundamentals of data science, machine learning, and deep learning. As a culmination of our learning journey, we embarked on a meaningful final project. Our project aims to contribute to the fight against online hate speech and harassment. We've developed a user-friendly program that empowers individuals to assess the offensiveness level of a given tweet and, in turn, generates an appropriate response. This tool is designed to promote a safer and more inclusive online environment.")

# Presentation file and button label
presentation_file = 'your_presentation.pdf' # Add the path to your presentation file
presentation_button_label = "Download Project Presentation"

# Create the button
if st.button(presentation_button_label):
    presentation_base64 = get_base64_of_bin_file(presentation_file)
    st.markdown(f'<a href="data:application/pdf;base64,{presentation_base64}" download="{presentation_file}">{presentation_button_label}</a>', unsafe_allow_html=True)

#########################################


# Part 2: Tweet Analysis
st.header("1️⃣ Analyze a Tweet")
st.markdown("For this first step, our goal is to predict the offensive nature of a tweet. To achieve this, we are using a BERT (tiny) model that we have trained and fine-tuned on a dataset of over 100,000 tweets (analyzed by humans and sorted as offensive or non-offensive).")

tweet = st.text_area(
     "Write down your tweet")

#if tweet:
    #st.write("You entered:")
    #st.write(tweet)

    # Add a button to check the tweet
if st.button("Check Tweet"):
    if tweet:
        # Replace this with your API call to check for offensiveness
        response = predict_binary(tweet)  # Replace with your API call
        st.write(response["type of tweet"])

        # If the API response is "the tweet is offensive," show the "Learn more" button
        # if response == "the tweet is offensive":
        #     if st.button("Learn more on my tweet"):
        #     # Scroll to Part 3
        #         st.write(
        #             "<a href='#tweet-description'>Scroll down to learn more about the tweet.</a>", unsafe_allow_html=True)
    else:
        st.write("Please enter a tweet to check.")


############################################

# Part 3: Tweet Description
st.header("2️⃣ Tweet Description")
st.markdown("For this second step, the objective is to classify the offensiveness of the tweet: Is the tweet racist? Homophobic? Misogynistic? Xenophobic? Transphobic? Does it target religion? To achieve this, we trained a GRU model on a dataset of over 40,000 tweets, which was also manually sorted by humans.")


if st.button("Analyze Tweet"):

    # Add code here to call your second API to describe the tweet
    response_class = predict_classif(tweet)
    description_result = f"This tweet has been analyzed as a {response_class['label']}"  # Replace with actual result
    st.write(description_result)

()
############################################
# Part 4: Generate Response to Tweet
st.header("3️⃣ Generate Response to Tweet")
st.markdown("Finally, for this last step, we propose to generate a response tweet, taking into account its offensive nature and the category of offense. To do so, we leveraged the OpenAI ChatGPT API, using an adapted prompt that incorporates the responses from the previous models.")




if st.button("Generate Response to Tweet"):
    # Add code here to call your third API to generate a response
    response_class = predict_classif(tweet)

    openai.api_key = os.environ.get("API_KEY")
    content_of_the_request = f"We have received an offensive tweet. This tweet can be classified as {response_class['label']}. Please find here the tweet '{tweet}'. Could you please generate a response to this tweet by explaining that this tweet is {response_class['label']} and recall the potential penalties incurred (legally but also in terms of banning on the tweeter platform). Please generate a response in the form of a tweet of max 280 characters and directly generate the quoted response without anything else."
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role':'user','content': content_of_the_request}])
    st.write(response.choices[0].message.content)


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
    team_member("IMG_0606.png", "Marianne Tatard", " ")

# Team Member 2
with col2:
    team_member("IMG_0607.png", "Lua-Marina Destailleurs", " ")

# Team Member 3
with col3:
    team_member("PP_Iris.png", "Iris Maréchal", " ")
