# gradio-app/app.py
import gradio as gr
import uuid
import requests
import time
from typing import Optional

# Fixed UUIDs from your database
USER1_ID = uuid.UUID('12345678-1234-5678-1234-567812345678')
USER2_ID = uuid.UUID('87654321-4321-8765-4321-876543210987')

# API configuration
API_BASE_URL = "http://localhost:8000"

# User data mapping
USERS = {
    "User1": {
        "id": USER1_ID,
        "username": "user1",
        "name": "User 1",
        "avatar": "👤"
    },
    "User2": {
        "id": USER2_ID,
        "username": "user2",
        "name": "User 2",
        "avatar": "👥"
    }
}

class RAGChatApp:
    def __init__(self):
        self.current_user_id = None
        self.current_user_name = None
        self.current_conversation_id = None
        self.conversations = {}
        self.messages = []
    
    def login(self, user_choice: str):
        """Login user"""
        if user_choice == "User 1":
            self.current_user_id = USER1_ID
            self.current_user_name = "User1"
        else:
            self.current_user_id = USER2_ID
            self.current_user_name = "User2"
        
        # Load user's conversations
        self.load_user_conversations()
        
        return f"✅ Logged in as: {user_choice}", gr.update(choices=list(self.conversations.keys()) if self.conversations else [])
    
    def load_user_conversations(self):
        """Load user's conversations from API"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/conversation/user/{str(self.current_user_id)}"
            )
            if response.status_code == 200:
                convs = response.json()
                self.conversations = {c.get("title", f"Conv {str(c['id'])[:8]}"): c["id"] for c in convs}
        except Exception as e:
            print(f"Error loading conversations: {e}")
    
    def create_new_conversation(self):
        """Create a new conversation"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/conversation/new",
                json={"user_id": str(self.current_user_id)}
            )
            if response.status_code == 201:
                data = response.json()
                self.current_conversation_id = str(data.get("conversation_id"))
                self.messages = []
                self.load_user_conversations()
                return f"✅ New conversation created!", gr.update(choices=list(self.conversations.keys()), value=None), []
            else:
                return f"❌ Failed to create conversation", gr.update(), []
        except Exception as e:
            return f"❌ Error: {e}", gr.update(), []
    
    def load_conversation(self, conv_title: str):
        """Load an existing conversation"""
        if not conv_title:
            return []
        
        conv_id = self.conversations.get(conv_title)
        if not conv_id:
            return []
        
        try:
            response = requests.get(f"{API_BASE_URL}/conversation/{conv_id}")
            if response.status_code == 200:
                data = response.json()
                dialogues = data.get("dialogues", [])
                self.current_conversation_id = conv_id
                self.messages = []
                chat_history = []
                
                for dialogue in dialogues:
                    prompt = dialogue.get("prompt", "")
                    answer = dialogue.get("answer", "")
                    if prompt:
                        chat_history.append((prompt, answer))
                    elif answer:
                        chat_history.append(("", answer))
                
                return chat_history
        except Exception as e:
            print(f"Error loading conversation: {e}")
        return []
    
    def send_message(self, message: str, history):
        """Send message and get response"""
        if not self.current_conversation_id:
            yield history + [("System", "Please create or select a conversation first")]
            return
        
        if not message:
            yield history
            return
        
        # Add user message to history
        history = history or []
        history.append((message, None))
        yield history
        
        try:
            # Send to API
            response = requests.post(
                f"{API_BASE_URL}/conversation/{self.current_conversation_id}/dialogue",
                json={"prompt": message}
            )
            
            if response.status_code == 201:
                dialogue_id = response.json().get("dialogue_id")
                
                # Poll for answer
                max_attempts = 30
                for _ in range(max_attempts):
                    time.sleep(1)
                    conv_response = requests.get(f"{API_BASE_URL}/conversation/{self.current_conversation_id}")
                    if conv_response.status_code == 200:
                        data = conv_response.json()
                        dialogues = data.get("dialogues", [])
                        for d in dialogues:
                            if str(d.get("id")) == dialogue_id and d.get("answer"):
                                # Update history with answer
                                history[-1] = (message, d.get("answer"))
                                yield history
                                return
                
                # Timeout
                history[-1] = (message, "⏰ Timeout waiting for response")
                yield history
            else:
                history[-1] = (message, f"❌ Error: {response.text}")
                yield history
                
        except Exception as e:
            history[-1] = (message, f"❌ Error: {e}")
            yield history

# Create app instance
app = RAGChatApp()

# Build Gradio interface
with gr.Blocks(title="RAG Demo App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 RAG Demo Application")
    gr.Markdown("Chat with AI about your documents")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 👤 User Selection")
            user_dropdown = gr.Dropdown(
                choices=["User 1", "User 2"],
                label="Select User",
                value=None
            )
            login_btn = gr.Button("Login", variant="primary")
            login_status = gr.Markdown("⚠️ Please login to start")
            
            gr.Markdown("---")
            gr.Markdown("### 💬 Conversations")
            
            refresh_btn = gr.Button("🔄 Refresh")
            conversation_dropdown = gr.Dropdown(
                choices=[],
                label="Select Conversation",
                interactive=True
            )
            load_conv_btn = gr.Button("📂 Load Conversation")
            new_conv_btn = gr.Button("✨ New Conversation", variant="primary")
            
            gr.Markdown("---")
            logout_btn = gr.Button("🚪 Logout")
        
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500
            )
            msg = gr.Textbox(
                label="Type your message here...",
                placeholder="Ask a question about your documents...",
                lines=2
            )
            clear = gr.Button("Clear Chat")
    
    # Event handlers
    login_btn.click(
        app.login,
        inputs=[user_dropdown],
        outputs=[login_status, conversation_dropdown]
    )
    
    new_conv_btn.click(
        app.create_new_conversation,
        inputs=[],
        outputs=[login_status, conversation_dropdown, chatbot]
    )
    
    load_conv_btn.click(
        app.load_conversation,
        inputs=[conversation_dropdown],
        outputs=[chatbot]
    )
    
    refresh_btn.click(
        app.load_user_conversations,
        inputs=[],
        outputs=[conversation_dropdown]
    )
    
    msg.submit(
        app.send_message,
        inputs=[msg, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "", None, [msg]
    )
    
    clear.click(lambda: [], None, chatbot)
    
    logout_btn.click(
        lambda: [None, None, "⚠️ Please login to start", gr.update(choices=[]), []],
        None,
        [user_dropdown, user_dropdown, login_status, conversation_dropdown, chatbot]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )