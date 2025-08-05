import streamlit as st
from transformers import pipeline
from huggingface_hub import login
from prompts import prompt_map
import os

from dotenv import load_dotenv
load_dotenv()


hf_token = os.environ.get("HF_TOKEN")
login(hf_token)


flan_generator = pipeline("text2text-generation", model="google/flan-t5-base")
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def generate_question(prompt):
    return flan_generator(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text']

def analyze_answer(answer):
    if len(answer.strip().split()) < 5:
        return "⚠️ Your answer is too short. Add more context."
    result = sentiment_analyzer(answer)[0]
    label = result['label']
    if label == "LABEL_2":
        return "✅ Confident and positive tone. Great job!"
    elif label == "LABEL_1":
        return "😐 Neutral tone. Add enthusiasm or specifics."
    elif label == "LABEL_0":
        return "⚠️ Negative tone. Try to show learning or resolution."
    else:
        return "🤖 Feedback unclear. Try rephrasing."

st.title("🧠 Smart Interview Practice Bot")

category = st.selectbox("Choose question type:", ["Behavioral", "Technical", "Tips"])
index = st.number_input("Question Number (1 to 3)", min_value=1, max_value=3, value=1)

if st.button("Ask Me a Question"):
    prompt = prompt_map[category][index - 1]
    question = generate_question(prompt)
    st.session_state["question"] = question

if "question" in st.session_state:
    st.subheader("AI Generated Question:")
    st.write(st.session_state["question"])

    if category != "Tips":
        answer = st.text_area("Your Answer:", key="user_answer")
        if st.button("Get Feedback"):
            feedback = analyze_answer(answer)
            st.success(feedback)
    else:
        st.info("No input required for tips.")
