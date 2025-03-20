import streamlit as st  # Importing Streamlit for web app creation
from openai import OpenAI  # Importing the OpenAI library to interact with the API

# Set up the page
st.set_page_config(page_title="Custom OpenAI Chat", page_icon="🤖")  # Configure the page title and icon
st.title("Custom OpenAI Chat Interface")  # Display the main title

# Initialize chat history
# Store the chat messages in the session state to persist across user interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration for API and model settings
with st.sidebar:
    st.header("API Configuration")  # Sidebar header for API settings
    
    # Input for API Base URL
    api_base = st.text_input(
        "API Base URL:",  # Label for Base URL input
        value="http://dev_litellm.myhomelab.org",  # Default value
        placeholder="http://dev_litellm.myhomelab.org"  # Hint text
    )  
    # NOTE: Make sure `api_base` is formatted appropriately for OpenAI or LiteLLM. OpenAI’s default base URL typically starts with "https://api.openai.com".
    # Example: OpenAI format: "https://api.openai.com/v1"

    # Input for API Key
    api_key = st.text_input("API Key:", type="password")  # Accept API key securely as password input
    # NOTE: The `api_key` should align with the platform being used:
    # - For OpenAI: The key must start with `sk-` (e.g., "sk-xxxxxxxxxxxx").
    # - For LiteLLM: Ensure the key is in the appropriate format expected by LiteLLM.

    # Dropdown for model selection
    selected_model = st.selectbox(
        "Choose Model",  # Label for model selection
        ("fake-1", "gpt-4o"),  # Available model options
        index=0  # Default selection index
    )
    
    # Button to reset the chat
    if st.button("Reset Chat"):
        st.session_state.messages = []  # Clear chat history

# Initialize OpenAI client with custom API settings
client = None  # Placeholder for the OpenAI client
if api_base and api_key:  # Ensure Base URL and API key are provided
    client = OpenAI(
        base_url=api_base,  # Set the API base URL
        api_key=api_key,  # Set the API key
    )

# Display chat messages from the session state
for message in st.session_state.messages:
    # Render each message in the appropriate role (user/assistant)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])  # Render the message content

# Handle user input via the chat input box
if prompt := st.chat_input("Type your message..."):  # User input box with prompt text
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user message instantly
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        # Check if the API client is properly configured
        if not client:
            st.error("⚠️ Please configure API settings in the sidebar!")  # Warn user if API settings are missing
            st.stop()  # Stop execution
        
        response_placeholder = st.empty()  # Placeholder to display streaming response
        full_response = ""  # Variable to hold the full assistant response
        
        try:
            # Generate streaming response from OpenAI API
            stream = client.chat.completions.create(
                model=selected_model,  # Use the selected model
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages  # Format chat history
                ],
                stream=True,  # Enable response streaming
            )
            
            # Stream the response chunk by chunk
            for chunk in stream:
                # Append non-empty content from the current chunk
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content  # Append chunk to full response
                    response_placeholder.markdown(full_response + "▌")  # Display streaming updates
                    
            response_placeholder.markdown(full_response)  # Render the final response
            
        except Exception as e:
            st.error(f"🚨 API Error: {str(e)}")  # Display error message for API issues
            if "geographic" in str(e).lower():  # Check for region-specific errors
                st.info("🌐 Try using a VPN connection to supported regions")  # Suggest using a VPN
            st.stop()  # Stop execution
        
        # Update the assistant's message in the chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})