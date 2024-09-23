# SentimentSage: Advanced Mental Health Chatbot with Emotion-Aware Responses

## Stack Up Bounty Submission: Llama Chatbot with Sentiment Analysis Integration

SentimentSage is a mental health assistant chatbot that integrates the Llama 2 language model with sentiment analysis to provide empathetic, context-aware responses. This project was developed as a submission for the Stack Up Bounty challenge "Llama Chatbot with Sentiment Analysis Integration".

### Features

- Sentiment analysis of user input to tailor responses
- Dynamic response generation using Llama 2 model
- Empathetic communication based on detected sentiment
- Contextual resource suggestions
- Conversation history tracking
- Continuous learning through dataset updates

### Prerequisites

- Python 3.7 or higher
- A Hugging Face account and API token

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/Felixdiamond/StackUp_Solutions.git
   cd StackUp_Solutions/SentimentSage
   ```

2. Install the required packages:
   ```
   pip install -q transformers huggingface_hub torch pandas gradio protobuf sentencepiece accelerate git+https://github.com/huggingface/transformers
   ```

3. Set up your Hugging Face API token as an environment variable:
   ```
   export HUGGINGFACE_TOKEN=your_token_here
   ```

### Usage

1. Run the main script:
   ```
   python sentimentsage.py
   ```

2. Open the Gradio interface link that appears in the console.

3. Start chatting with SentimentSage in the web interface!

### Project Structure

- `sentimentsage.py`: Main script containing the chatbot logic
- `mental_health_qa_dataset.csv`: Dataset for storing Q&A pairs and sentiment
- `README.md`: This file

### How It Works

1. User input is analyzed for sentiment using the Hugging Face sentiment analysis pipeline.
2. The Llama 2 model generates a contextual response based on the input and detected sentiment.
3. Empathetic statements and resource suggestions are added based on the sentiment.
4. The response is displayed to the user, and the interaction is stored in the dataset.

### Evaluation Criteria Addressed

1. Integration of Sentiment Analysis: Effectively uses Hugging Face's sentiment analysis pipeline to analyze user input.
2. Enhanced Chatbot Functionality: Adjusts responses based on detected sentiment, providing empathetic and contextual communication.
3. Creativity and Originality: Implements a mental health focus with dynamic resource suggestions and empathetic responses.
4. Code Quality: Well-structured, commented code with clear function definitions and logic flow.
5. Documentation: Comprehensive README and in-code comments explaining functionality.
6. Practical Application: Directly applicable in mental health support scenarios.