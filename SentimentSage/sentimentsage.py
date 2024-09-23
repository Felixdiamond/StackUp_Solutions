import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import torch
import gradio as gr
import random

# Hugging Face login
login(token='your_token')

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Define the path for the CSV file
csv_file = 'mental_health_qa_dataset.csv'

# Check if the CSV file exists; if not, create it with initial data
if not os.path.exists(csv_file):
    qa_data = {
        'question': ["How can I manage stress?", "What are signs of depression?"],
        'answer': ["Some ways to manage stress include deep breathing, exercise, and mindfulness practices.", 
                   "Common signs of depression include persistent sadness, loss of interest in activities, and changes in sleep patterns."],
        'sentiment': ["NEUTRAL", "NEGATIVE"]
    }
    qa_df = pd.DataFrame(qa_data)
    qa_df.to_csv(csv_file, index=False)
else:
    # Load the existing CSV file into a DataFrame
    qa_df = pd.read_csv(csv_file)

# Initialize the Llama 2 model and tokenizer
model_id = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.use_default_system_prompt = False

# Initialize the pipeline using Hugging Face pipeline
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length=1024,
)

# Memory to keep conversation history
conversation_history = []

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

def get_empathetic_response(sentiment):
    responses = {
        "POSITIVE": [
            "I'm glad you're feeling positive! That's a great foundation for mental health.",
            "It's wonderful to hear you're in a good mood. Let's build on that positivity.",
            "Your positive attitude is inspiring. How can we maintain this good energy?"
        ],
        "NEGATIVE": [
            "I'm here to support you through these challenging feelings.",
            "It's okay to feel down sometimes. Let's work through this together.",
            "I'm sorry you're feeling this way. Remember, it's temporary and there is hope."
        ],
        "NEUTRAL": [
            "I'm here to listen and help in any way I can.",
            "Feel free to share more about what's on your mind.",
            "Let's explore your thoughts and feelings together."
        ]
    }
    return random.choice(responses[sentiment])

def suggest_resources(sentiment):
    resources = {
        "POSITIVE": [
            "For more positive reinforcement, try the Happify app.",
            "Consider reading 'The Happiness Advantage' by Shawn Achor to build on your positive mindset."
        ],
        "NEGATIVE": [
            "The Crisis Text Line (text HOME to 741741) is available 24/7 for support.",
            "The app 'Calm Harm' can be helpful for managing difficult emotions."
        ],
        "NEUTRAL": [
            "The Headspace app offers great mindfulness exercises.",
            "Consider checking out the book 'Feeling Good' by David Burns for general mental wellness tips."
        ]
    }
    return random.choice(resources[sentiment])

def generate_mental_health_response(question, sentiment):
    # Generate response considering previous conversation history
    conversation_context = "\n".join([f"User: {q}\nBot: {a}" for q, a in conversation_history[-10:]])
    
    prompt = f"""As a mental health assistant, provide a compassionate and helpful response. Here is the previous conversation for context:

{conversation_context}

The user's current emotional state is {sentiment.lower()}.

Question: {question}

Response:"""
    
    response = llama_pipeline(prompt, max_length=300, do_sample=True)[0]['generated_text']
    return response.split("Response:")[-1].strip()

def answer_question(question):
    global qa_df
    
    # Analyze sentiment of the question
    sentiment, confidence = analyze_sentiment(question)
    
    # Get an empathetic response based on sentiment
    empathetic_response = get_empathetic_response(sentiment)
    
    # Generate a mental health-focused response
    main_response = generate_mental_health_response(question, sentiment)
    
    # Suggest a resource
    resource_suggestion = suggest_resources(sentiment)
    
    # Combine all parts of the response
    full_response = f"{empathetic_response}\n\n{main_response}\n\nHere's a helpful resource: {resource_suggestion}"
    
    # Add the new QA pair to the dataset
    new_row = pd.DataFrame({'question': [question], 'answer': [full_response], 'sentiment': [sentiment]})
    qa_df = pd.concat([qa_df, new_row], ignore_index=True)
    qa_df.to_csv(csv_file, index=False)
    
    # Update conversation history
    conversation_history.append((question, full_response))
    
    return full_response


with gr.Blocks() as interface:
    gr.Markdown("# Mental Health Assistant Chatbot")
    gr.Markdown("This chatbot uses sentiment analysis to provide personalized mental health support.")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    feedback = gr.Radio(["👍", "👎"], label="Rate this response")
    clear = gr.Button("Clear")
    
    def user(user_message, history):
        response = answer_question(user_message)
        return "", history + [(user_message, response)]

    def feedback_received(feedback, history):
        # Handle feedback submission (e.g., log it for further analysis)
        print(f"Feedback received: {feedback}")
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    feedback.change(feedback_received, [feedback, chatbot], [chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio Interface
interface.launch()