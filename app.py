import random
import pandas as pd
import streamlit as st
import time
from sentence_transformers import SentenceTransformer, util

# Load the dataset
df = pd.read_csv('Software_Questions.csv', encoding='ISO-8859-1')

# Extract necessary columns
topics = df['Category'].unique()
questions = df['Question'].tolist()
answers = df['Answer'].tolist()
categories = df['Category'].tolist()

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize session state for progress tracking
if 'correct_count' not in st.session_state:
    st.session_state['correct_count'] = 0
if 'attempt_count' not in st.session_state:
    st.session_state['attempt_count'] = 0
if 'score' not in st.session_state:
    st.session_state['score'] = 0
if 'incorrect_questions' not in st.session_state:
    st.session_state['incorrect_questions'] = []
if 'question_attempts' not in st.session_state:
    st.session_state['question_attempts'] = 0
if 'user_answer' not in st.session_state:
    st.session_state['user_answer'] = ''

# Streamlit UI
st.title("Interactive Job Interview Practice Bot")
st.write("Select a topic, answer the question, and get feedback to improve your skills!")

# Topic and Difficulty Selection
selected_topic = st.selectbox("Choose a Topic:", topics)
difficulty = st.selectbox("Select Difficulty Level:", ["Easy", "Medium", "Hard"])

# Question Generation
if selected_topic:
    # Filter questions based on the selected topic
    topic_questions_df = df[df['Category'] == selected_topic]

    if st.button("Generate Question"):
        # Randomly select a question from the filtered topic
        question_idx = random.choice(topic_questions_df.index)
        question_text = topic_questions_df.loc[question_idx, 'Question']
        correct_answer = topic_questions_df.loc[question_idx, 'Answer']

        # Start the timer and reset question attempts and user answer
        st.session_state['start_time'] = time.time()
        st.session_state['question_attempts'] = 0
        st.session_state['user_answer'] = ''  # Clear previous answer input

        # Store question and answer in session state
        st.session_state['current_question'] = question_text
        st.session_state['correct_answer'] = correct_answer

    # Display the generated question
    if 'current_question' in st.session_state:
        st.write("*Question:*", st.session_state['current_question'])

        # Get User's Answer with a fresh input box each time
        user_answer = st.text_input("Your Answer:", value=st.session_state['user_answer'])
        st.session_state['user_answer'] = user_answer  # Save user input to session state

        # Feedback
        if st.button("Submit Answer"):
            # Calculate the time taken to answer
            time_taken = time.time() - st.session_state['start_time']
            st.write(f"⏱️ Time taken: {time_taken:.2f} seconds")

            # Compare the user answer with the correct answer using semantic similarity
            user_embedding = model.encode(user_answer)
            correct_embedding = model.encode(st.session_state['correct_answer'])
            similarity_score = util.pytorch_cos_sim(user_embedding, correct_embedding).item()

            # Update attempt count for the current question
            st.session_state['question_attempts'] += 1
            st.session_state['attempt_count'] += 1

            # Feedback based on similarity score
            if similarity_score > 0.85:
                st.write("🎉 Great job! That's fully correct! 👍")
                st.session_state['correct_count'] += 1
                st.session_state['score'] += int(similarity_score * 100)  # Add to score
            elif 0.6 < similarity_score <= 0.85:
                st.write("👍 Nice attempt! Your answer is partially correct.")
                st.session_state['score'] += int(similarity_score * 50)  # Partial score
                if st.session_state['question_attempts'] == 2:
                    st.write("💡 Here’s the complete answer for reference:")
                    st.write("*Answer:*", st.session_state['correct_answer'])
                    st.session_state['incorrect_questions'].append((st.session_state['current_question'], st.session_state['correct_answer']))
            else:
                # First attempt: Show a hint
                if st.session_state['question_attempts'] == 1:
                    hint = " ".join(st.session_state['correct_answer'].split()[:3])
                    st.write(f"Hint: Think about - {hint}...")
                    st.write("Try again!")
                elif st.session_state['question_attempts'] == 2:
                    st.write("💡 Not quite. Here’s the correct answer:")
                    st.write("*Answer:*", st.session_state['correct_answer'])
                    st.session_state['incorrect_questions'].append((st.session_state['current_question'], st.session_state['correct_answer']))

            # Display progress
            st.write(f"Progress: {st.session_state['correct_count']} correct out of {st.session_state['attempt_count']} attempts.")
            st.write(f"Your current score: {st.session_state['score']}")

        # Review Incorrect Questions
        if st.button("Review Incorrect Questions"):
            if st.session_state['incorrect_questions']:
                st.write("Review your incorrect answers:")
                for question, answer in st.session_state['incorrect_questions']:
                    st.write(f"❌ *Question:* {question}")
                    st.write(f"*Correct Answer:* {answer}")
            else:
                st.write("No incorrect questions to review. Great job!")
                
        # Reset
        if st.button("Reset Progress"):
            st.session_state['correct_count'] = 0
            st.session_state['attempt_count'] = 0
            st.session_state['score'] = 0
            st.session_state['incorrect_questions'] = []
            st.session_state['question_attempts'] = 0
            st.session_state['user_answer'] = ''
            if 'current_question' in st.session_state:
                del st.session_state['current_question']
            if 'correct_answer' in st.session_state:
                del st.session_state['correct_answer']
            st.write("Progress has been reset. Start fresh!")
