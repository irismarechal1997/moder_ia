import streamlit as st

# Modification des paramètres visuels du site
primaryColor = "#F63366"
textColor = "#262730"
font = "sans serif"

# Ajout d'une photo dans le fond du site
def set_background():
    st.markdown(
        f"""
        <style>
        .stApp {
            background-image: url("irismarechal1997/moder_ia/data/Background_site.png");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Page 1: Fighting against online hate speech
def page_one():
    set_background()
    st.title("Fighting online hate-speech")

    # Add a text input field for entering a tweet
    tweet = st.text_area("Write down your tweet")

    # Display the entered tweet
    if tweet:
        st.write("You entered:")
        st.write(tweet)

    # Add a button to check the tweet
    if st.button("Check Tweet"):
        if tweet:
            # Replace this with your API call to check for offensiveness
            response = check_tweet_offensiveness(tweet)
            st.write("API Response:", response)

            # If the API response is "the tweet is offensive," show the "Learn more" button
            if response == "the tweet is offensive":
                st.button("Learn more on my tweet")

# Page 2: Learn more on my tweet
def page_two():
    set_background()
    st.title("Learn more on my tweet")

# Simulate an API call to check tweet offensiveness
def check_tweet_offensiveness(tweet):
    # Replace this with your actual API call
    # For simulation purposes, we'll assume it's offensive if it contains the word "hate"
    if "hate" in tweet.lower():
        return "the tweet is offensive"
    else:
        return "the tweet is not offensive"

# Main function to run the Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to:", ["Home", "Learn More"])

    if page == "Home":
        page_one()
    elif page == "Learn More":
        page_two()

if __name__ == "__main__":
    main()
