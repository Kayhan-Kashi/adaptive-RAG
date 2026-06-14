import streamlit as st
import requests
import uuid

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"  # Update this to your FastAPI server URL
USER1_ID = '12345678-1234-5678-1234-567812345678'
USER2_ID = '87654321-4321-8765-4321-876543210987'

st.set_page_config(page_title="Conversation App", layout="wide")

# --- Sidebar: User Selection ---
st.sidebar.title("Configuration")
selected_user = st.sidebar.selectbox(
    "Select Current User",
    options=[USER1_ID, USER2_ID],
    format_func=lambda x: "User 1" if x == USER1_ID else "User 2"
)

# --- Main App Logic ---
st.title("Conversation Manager")

# 1. Fetch Conversations
def get_user_conversations(user_id):
    response = requests.get(f"{API_BASE_URL}/conversation/user/{user_id}")
    return response.json() if response.status_code == 200 else []

# 2. Create Conversation
def create_new_conversation(user_id):
    response = requests.post(
        f"{API_BASE_URL}/conversation/new", 
        json={"user_id": user_id}
    )
    return response.json()

# 3. Get Dialogues
def get_dialogues(conversation_id):
    response = requests.get(f"{API_BASE_URL}/conversation/{conversation_id}/dialogues")
    return response.json() if response.status_code == 200 else []

# 4. Add Dialogue
def add_dialogue(conversation_id, prompt):
    response = requests.post(
        f"{API_BASE_URL}/conversation/{conversation_id}/dialogue",
        json={"prompt": prompt}
    )
    return response.json()

# --- UI Flow ---
conversations = get_user_conversations(selected_user)

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Your Conversations")
    if st.button("Start New Conversation"):
        new_conv = create_new_conversation(selected_user)
        st.rerun()
    
    # List conversations
    conv_options = {c['id']: f"Conv: {c['id'][:8]}..." for c in conversations}
    selected_conv_id = st.radio("Select a conversation", options=list(conv_options.keys()), format_func=lambda x: conv_options[x])

with col2:
    if selected_conv_id:
        st.header(f"Conversation: {selected_conv_id}")
        
        # Display Dialogues
        dialogues = get_dialogues(selected_conv_id)
        for d in dialogues:
            st.chat_message("user").write(d.get('prompt', ''))
            if d.get('answer'):
                st.chat_message("assistant").write(d.get('answer'))
        
        # Input for new message
        if prompt := st.chat_input("What is your question?"):
            add_dialogue(selected_conv_id, prompt)
            st.rerun()
    else:
        st.info("Select or create a conversation to start chatting.")
