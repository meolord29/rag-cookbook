# Simple Chatbot Implementation

This recipe demonstrates how to build a basic chatbot application using a Large Language Model (LLM) and Streamlit for the user interface.

## Questions This Recipe Answers
- How do I send and receive a response from my LLM model?
- Should I use chat completion or chat streaming?
- How do I make a simple chatbot app?

## Key Components

### 1. LLM Integration
The recipe shows how to connect to an LLM API (like OpenAI's GPT models) to generate responses based on user inputs.

**Implementation:**
```python
def get_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content
```

### 2. Chat Streaming
Implements streaming responses from the LLM to provide a more interactive user experience.

**Benefits of Streaming:**
- Provides immediate feedback to users
- Creates a more natural conversational experience
- Reduces perceived latency

**Implementation:**
```python
def get_streaming_response(messages, model="gpt-3.5-turbo"):
    response = ""
    for chunk in client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    ):
        content = chunk.choices[0].delta.content
        if content is not None:
            response += content
            yield response
```

### 3. User Interface with Streamlit
Builds a simple but effective chat interface using Streamlit's components.

**Features:**
- Message history tracking
- User input handling
- Styled message display
- Model selection options

## Implementation Details

This recipe demonstrates the fundamental building blocks of a chatbot application:

1. **API Connection**: Setting up the connection to the LLM provider
2. **Message Handling**: Managing the conversation history
3. **Response Generation**: Getting completions from the LLM
4. **User Interface**: Creating an intuitive chat interface

## Usage

To run the chatbot application:

1. Install the required dependencies:
   ```bash
   pip install openai streamlit
   ```

2. Set up your API key as an environment variable or in the application

3. Run the Streamlit app:
   ```bash
   streamlit run app/simple_chatbot.py
   ```

## References
1. [OpenAI Python Library](https://pypi.org/project/openai/) - Provides examples on how to generate responses
2. [OpenAI API Base URL Configuration](https://www.restack.io/p/openai-python-answer-change-base-url-cat-ai) - Shows how to change the API base if you use LiteLLM or any other custom base URL
3. [Streamlit Documentation](https://streamlit.io/) - Basics of Streamlit for front-end design
4. [Building an LLM-powered Chatbot with Streamlit](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/) - Offers details on how to make a simple front end for ChatGPT