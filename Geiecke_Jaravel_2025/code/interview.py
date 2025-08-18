import streamlit as st  # Imports Streamlit for building the web UI.
import time             # Imports time for timestamps and measuring durations.
from utils import (     # Imports helper functions for authentication and data management.
    check_password,
    check_if_interview_completed,
    save_interview_data,
)
import os               # Imports os for file and directory operations.
import config           # Imports configuration settings (API keys, model, directories, etc.).

# --- BEGIN: Added Ollama option for gemma3 model ---   

import requests         # Ollama API uses HTTP requests

# --- END: Added Ollama option for gemma3 model ---

# Load API library depending on model type.
if "gpt" in config.MODEL.lower():  # Checks if the model name contains 'gpt' (case-insensitive).
    api = "openai"                 # Sets API type to OpenAI.
    from openai import OpenAI      # Imports OpenAI client for API calls.

elif "claude" in config.MODEL.lower():  # Checks if the model name contains 'claude'.
    api = "anthropic"                  # Sets API type to Anthropic.
    import anthropic                   # Imports Anthropic client for API calls.

# --- BEGIN: Added Ollama option for gemma3 model ---

elif "gemma3" in config.MODEL.lower():
    api = "ollama"            # Sets API type to Ollama for local models.

# --- END: Added Ollama option for gemma3 model --

else:
    raise ValueError(                  # Raises error if model type is not recognized.
        "Model does not contain 'gpt', 'claude', or 'gemma3'; unable to determine API."
    )

# Set page title and icon for the Streamlit app.
st.set_page_config(page_title="Interview", page_icon=config.AVATAR_INTERVIEWER)

# Check if login functionality is enabled.
if config.LOGINS:
    # Display login form and check password.
    pwd_correct, username = check_password()  # Returns True and username if login is successful.
    if not pwd_correct:                       # If password is incorrect, stop the app.
        st.stop()
    else:
        st.session_state.username = username  # Store username in session state.
else:
    st.session_state.username = "testaccount" # Use default username if logins are disabled.

# Ensure required directories exist for storing transcripts and backups.
if not os.path.exists(config.TRANSCRIPTS_DIRECTORY):
    os.makedirs(config.TRANSCRIPTS_DIRECTORY)
if not os.path.exists(config.TIMES_DIRECTORY):
    os.makedirs(config.TIMES_DIRECTORY)
if not os.path.exists(config.BACKUPS_DIRECTORY):
    os.makedirs(config.BACKUPS_DIRECTORY)

# Initialize interview activity state in session.
if "interview_active" not in st.session_state:
    st.session_state.interview_active = True  # Interview is active by default.

# Initialize chat message history in session.
if "messages" not in st.session_state:
    st.session_state.messages = []            # Start with empty message list.

# Store interview start time and formatted timestamp in session.
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time() # Record current time as start.
    st.session_state.start_time_file_names = time.strftime(
        "%Y_%m_%d_%H_%M_%S", time.localtime(st.session_state.start_time)
    )  # Format start time for filenames.

# Check if the interview was already completed for this user.
interview_previously_completed = check_if_interview_completed(
    config.TIMES_DIRECTORY, st.session_state.username
)

# If interview was completed and no messages are present, mark as inactive.
if interview_previously_completed and not st.session_state.messages:
    st.session_state.interview_active = False
    completed_message = "Interview already completed."
    st.markdown(completed_message)  # Display completion message.

# Add a 'Quit' button to the dashboard using Streamlit columns.
col1, col2 = st.columns([0.85, 0.15])  # Layout: main area and small button area.
with col2:
    # If interview is active and user clicks 'Quit', end interview.
    if st.session_state.interview_active and st.button(
        "Quit", help="End the interview."
    ):
        st.session_state.interview_active = False  # Mark interview as inactive.
        quit_message = "You have cancelled the interview."
        st.session_state.messages.append({"role": "assistant", "content": quit_message})  # Log quit message.
        save_interview_data(
            st.session_state.username,
            config.TRANSCRIPTS_DIRECTORY,
            config.TIMES_DIRECTORY,
        )  # Save transcript and timing data.

# On rerun, display previous conversation (excluding system prompt/first message).
for message in st.session_state.messages[1:]:
    if message["role"] == "assistant":
        avatar = config.AVATAR_INTERVIEWER
    else:
        avatar = config.AVATAR_RESPONDENT
    # Only display messages that do not contain closing codes.
    if not any(code in message["content"] for code in config.CLOSING_MESSAGES.keys()):
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# Load the appropriate API client based on selected model.
if api == "openai":
    client = OpenAI(api_key=st.secrets["API_KEY_OPENAI"])  # Initialize OpenAI client.
    api_kwargs = {"stream": True}                          # Enable streaming responses.
elif api == "anthropic":
    client = anthropic.Anthropic(api_key=st.secrets["API_KEY_ANTHROPIC"])  # Anthropic client.
    api_kwargs = {"system": config.SYSTEM_PROMPT}                          # Pass system prompt.
# --- BEGIN: Added Ollama option for gemma3 model ---

elif api == "ollama":
    # Ollama uses HTTP requests, so no persistent client object is needed.
    api_kwargs = {
        "url": "http://localhost:11434/api/chat",
        "json": {
            "model": config.MODEL,
            "stream": False,
            "temperature": config.TEMPERATURE,
            "messages": [],
        }
    }

# --- END: Added Ollama option for gemma3 model ---

# Set API call parameters for chat completion.
api_kwargs["messages"] = st.session_state.messages
api_kwargs["model"] = config.MODEL
api_kwargs["max_tokens"] = config.MAX_OUTPUT_TOKENS
if config.TEMPERATURE is not None:
    api_kwargs["temperature"] = config.TEMPERATURE

# If no messages exist, send initial system prompt and display first interviewer message.
if not st.session_state.messages:
    if api == "openai":
        st.session_state.messages.append(
            {"role": "system", "content": config.SYSTEM_PROMPT}
        )  # Add system prompt to message history.
        with st.chat_message("assistant", avatar=config.AVATAR_INTERVIEWER):
            stream = client.chat.completions.create(**api_kwargs)  # Request first message from API.
            message_interviewer = st.write_stream(stream)          # Display streamed response.

    elif api == "anthropic":
        st.session_state.messages.append({"role": "user", "content": "Hi"})  # Anthropic expects user message first.
        with st.chat_message("assistant", avatar=config.AVATAR_INTERVIEWER):
            message_placeholder = st.empty()
            message_interviewer = ""
            with client.messages.stream(**api_kwargs) as stream:
                for text_delta in stream.text_stream:
                    if text_delta != None:
                        message_interviewer += text_delta
                    message_placeholder.markdown(message_interviewer + "▌")  # Show streaming text.
            message_placeholder.markdown(message_interviewer)                # Show final message.
    
    # --- BEGIN: Added Ollama option for gemma3 model ---

    elif api == "ollama":   
        st.session_state.messages.append(
            {"role": "system", "content": config.SYSTEM_PROMPT}
        )

        with st.chat_message("assistant", avatar=config.AVATAR_INTERVIEWER):
            response = requests.post(
                url=api_kwargs["url"],
                json={
                    "model": config.MODEL,
                    "messages": st.session_state.messages,
                    "stream": False,
                    "temperature": config.TEMPERATURE,
                    "max_tokens": config.MAX_OUTPUT_TOKENS,
                }
            )
            message_interviewer = response.json()["message"]["content"]

    # --- END: Added Ollama option for gemma3 model ---

    st.session_state.messages.append(
        {"role": "assistant", "content": message_interviewer}
    )  # Store interviewer message.

    # Save initial backup files to record interview start.
    save_interview_data(
        username=st.session_state.username,
        transcripts_directory=config.BACKUPS_DIRECTORY,
        times_directory=config.BACKUPS_DIRECTORY,
        file_name_addition_transcript=f"_transcript_started_{st.session_state.start_time_file_names}",
        file_name_addition_time=f"_time_started_{st.session_state.start_time_file_names}",
    )

# Main chat loop: only active if interview is ongoing.
if st.session_state.interview_active:
    # Display chat input for respondent.
    if message_respondent := st.chat_input("Your message here"):
        st.session_state.messages.append(
            {"role": "user", "content": message_respondent}
        )  # Add respondent's message to history.

        # Display respondent's message in chat.
        with st.chat_message("user", avatar=config.AVATAR_RESPONDENT):
            st.markdown(message_respondent)

        # Generate and display interviewer response.
        with st.chat_message("assistant", avatar=config.AVATAR_INTERVIEWER):
            message_placeholder = st.empty()  # Placeholder for streaming message.
            message_interviewer = ""          # Initialize interviewer message.

            if api == "openai":
                stream = client.chat.completions.create(**api_kwargs)  # Stream response from OpenAI.
                for message in stream:
                    text_delta = message.choices[0].delta.content
                    if text_delta != None:
                        message_interviewer += text_delta
                    # Display message after 5 characters to check for codes first.
                    if len(message_interviewer) > 5:
                        message_placeholder.markdown(message_interviewer + "▌")
                    if any(
                        code in message_interviewer
                        for code in config.CLOSING_MESSAGES.keys()
                    ):
                        message_placeholder.empty()  # Stop streaming if closing code detected.
                        break

            elif api == "anthropic":
                with client.messages.stream(**api_kwargs) as stream:
                    for text_delta in stream.text_stream:
                        if text_delta != None:
                            message_interviewer += text_delta
                        if len(message_interviewer) > 5:
                            message_placeholder.markdown(message_interviewer + "▌")
                        if any(
                            code in message_interviewer
                            for code in config.CLOSING_MESSAGES.keys()
                        ):
                            message_placeholder.empty()
                            break

            # --- BEGIN: Added Ollama option for gemma3 model ---

            elif api == "ollama":       
                response = requests.post(
                    url=api_kwargs["url"],
                    json={
                        "model": config.MODEL,
                        "messages": st.session_state.messages,
                        "stream": False,
                        "temperature": config.TEMPERATURE,
                        "max_tokens": config.MAX_OUTPUT_TOKENS,
                    }
                )

                message_interviewer = response.json()["message"]["content"]
                message_placeholder.markdown(message_interviewer + "▌")

            # --- END: Added Ollama option for gemma3 model ---

            # If no closing code, display and store interviewer message.
            if not any(
                code in message_interviewer for code in config.CLOSING_MESSAGES.keys()
            ):
                message_placeholder.markdown(message_interviewer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": message_interviewer}
                )
                # Save progress as backup, ignore errors.
                try:
                    save_interview_data(
                        username=st.session_state.username,
                        transcripts_directory=config.BACKUPS_DIRECTORY,
                        times_directory=config.BACKUPS_DIRECTORY,
                        file_name_addition_transcript=f"_transcript_started_{st.session_state.start_time_file_names}",
                        file_name_addition_time=f"_time_started_{st.session_state.start_time_file_names}",
                    )
                except:
                    pass

            # If closing code detected, display closing message and end interview.
            for code in config.CLOSING_MESSAGES.keys():
                if code in message_interviewer:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": message_interviewer}
                    )
                    st.session_state.interview_active = False
                    closing_message = config.CLOSING_MESSAGES[code]
                    st.markdown(closing_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": closing_message}
                    )
                    # Save final transcript and timing data, retry until successful.
                    final_transcript_stored = False
                    while final_transcript_stored == False:
                        save_interview_data(
                            username=st.session_state.username,
                            transcripts_directory=config.TRANSCRIPTS_DIRECTORY,
                            times_directory=config.TIMES_DIRECTORY,
                        )
                        final_transcript_stored = check_if_interview_completed(
                            config.TRANSCRIPTS_DIRECTORY, st.session_state.username
                        )
                        time.sleep(0.1)