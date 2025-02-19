from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# img2text using *Pipeline* (download locally)
#def img2text(url):
#    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base",max_new_tokens=50)

#    text = image_to_text(url)[0]['generated_text']

#    return text


# img2text using API
def img2text_API(url):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    with open(url, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
 
    return response.json()[0]['generated_text']


# llm - USING $$$$--OPENAI--$$$$$
def generate_story(scenario):
    template = """
    You are a story teller: 
    You can generate a short story based on a simple narrative, the story should be no more than 40 words;

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=ChatOpenAI(
        model_name="gpt-3.5-turbo",temperature=1), prompt=prompt, verbose=True)
    
    story = story_llm.predict(scenario=scenario)

    return story

# text to speech API
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)

def main():
    st.set_page_config(page_title="Image Story", page_icon=":tada:", layout="wide")
    st.header("Turn an Image into a Story")

    uploaded_file = None
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        #scenario = img2text(uploaded_file.name)
        scenario = img2text_API(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")


if __name__ == '__main__':
    main()