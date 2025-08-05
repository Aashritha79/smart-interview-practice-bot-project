import os
from transformers import pipeline
from huggingface_hub import login
from prompts import prompt_map

from dotenv import load_dotenv
load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
login(hf_token)



flan_generator = pipeline("text2text-generation", model="google/flan-t5-base")
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def show_intro():
    print("""
🔹 AI Interview Assistant 🔹

Hi! I'm here to help you practice for job interviews.
At any time, type 'exit' to end the session.
""")

def show_menu():
    print("""
Please select the type of questions you'd like to practice:
1. Behavioral Interview Questions
2. Technical Interview Questions
3. General Interview Tips
""")

def goodbye():
    print("\nGoodbye! 👋 Stay confident, and keep practicing!\n")

def ask_question(prompt, flan_generator):
    response = flan_generator(prompt, max_new_tokens=100, do_sample=False)
    return response[0]['generated_text']

def analyze_answer(answer):
    word_count = len(answer.strip().split())
    if word_count < 5:
        return "Your answer is too short. Try to include more detail or context."
    result = sentiment_analyzer(answer)[0]
    label = result['label']
    if label == "LABEL_2":
        return "✅ Your response has a confident and positive tone. Good job!"
    elif label == "LABEL_1":
        return "😐 Your response feels neutral. Try adding a bit more enthusiasm or personal reflection."
    elif label == "LABEL_0":
        return "⚠️ Your tone seems negative. In interviews, aim to show growth or resolution even in tough situations."
    else:
        return "Feedback not clear. Try rephrasing your answer."

show_intro()
show_menu()

valid_choices = {"1": "Behavioral", "2": "Technical", "3": "Tips"}

while True:
    choice = input("Your choice (1/2/3): ").strip()
    if choice in valid_choices:
        selected_type = valid_choices[choice]
        break
    elif choice.lower() == "exit":
        goodbye()
        exit()
    else:
        print("Invalid input. Please enter 1, 2, or 3.")

print(f"\n🔹 Great! Let’s start with {selected_type} questions.\n")

for i, prompt in enumerate(prompt_map[selected_type], start=1):
    print(f"\n🔸 Question {i}:")
    question = ask_question(prompt, flan_generator)
    print("AI:", question)

    if selected_type == "Tips":
        continue

    user_answer = input("Your answer: ")
    if user_answer.lower() == "exit":
        goodbye()
        exit()

    feedback = analyze_answer(user_answer)
    print("Feedback:", feedback)

print("\n✅ Session complete! Great job practicing. 🚀")