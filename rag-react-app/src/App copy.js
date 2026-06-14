// src/App.js
import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './App.css';

// Fixed UUIDs from your database
const USER1_ID = '12345678-1234-5678-1234-567812345678';
const USER2_ID = '87654321-4321-8765-4321-876543210987';

const API_BASE_URL = 'http://localhost:8000';
const WS_BASE_URL = 'ws://localhost:8000';

const USERS = {
  'User1': { id: USER1_ID, username: 'user1', name: 'User1', avatar: '👤' },
  'User2': { id: USER2_ID, username: 'user2', name: 'User2', avatar: '👥' }
};

function App() {
  const [user, setUser] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentView, setCurrentView] = useState('login');
  const [wsConnected, setWsConnected] = useState(false);
  
  const wsRef = useRef(null);
  const messageQueueRef = useRef([]);

  const handleWebSocketMessage = useCallback((data) => {
    const { type } = data;
    
    if (type === 'answer') {
      const { conversation_id, answer } = data;
      if (conversation_id === currentConversation?.id) {
        setLoading(false);
        setMessages(prev => [...prev, { role: 'assistant', content: answer }]);
      }
    } else if (type === 'ack') {
      console.log('Message acknowledged');
    } else if (type === 'error') {
      console.error('Server error:', data.error);
      setLoading(false);
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${data.error}` }]);
    }
  }, [currentConversation?.id]);

  const connectWebSocket = useCallback(() => {
    if (!user) return;
    
    const wsUrl = `${WS_BASE_URL}/ws/${user.id}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setWsConnected(true);
      
      // Send any queued messages
      while (messageQueueRef.current.length > 0) {
        const msg = messageQueueRef.current.shift();
        ws.send(JSON.stringify(msg));
      }
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsConnected(false);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWsConnected(false);
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        if (user && currentView === 'chat') {
          connectWebSocket();
        }
      }, 3000);
    };
    
    wsRef.current = ws;
  }, [user, currentView, handleWebSocketMessage]);

  // WebSocket connection
  useEffect(() => {
    if (user && currentView === 'chat') {
      connectWebSocket();
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [user, currentView, connectWebSocket]);

  const sendWebSocketMessage = useCallback((conversationId, prompt) => {
    const message = {
      type: 'chat',
      conversation_id: conversationId,
      prompt: prompt
    };
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      // Queue message for when connection is established
      messageQueueRef.current.push(message);
    }
  }, []);

  const loadConversations = useCallback(async () => {
    if (!user) return;
    try {
      const response = await axios.get(`${API_BASE_URL}/conversation/user/${user.id}`);
      setConversations(response.data);
    } catch (error) {
      console.error('Error loading conversations:', error);
    }
  }, [user]);

  useEffect(() => {
    if (user) {
      loadConversations();
    }
  }, [user, loadConversations]);

  const handleLogin = (userKey) => {
    const selectedUser = USERS[userKey];
    setUser(selectedUser);
    setCurrentView('dashboard');
  };

  const handleLogout = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setUser(null);
    setConversations([]);
    setCurrentConversation(null);
    setMessages([]);
    setWsConnected(false);
    setCurrentView('login');
  };

  const createNewConversation = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/conversation/new`, {
        user_id: user.id
      });
      const newConversation = response.data;
      setCurrentConversation({ id: newConversation.conversation_id });
      setMessages([]);
      setCurrentView('chat');
      await loadConversations();
    } catch (error) {
      console.error('Error creating conversation:', error);
    }
  };

  const loadConversation = async (conversationId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/conversation/${conversationId}`);
      const conversation = response.data;
      setCurrentConversation({ id: conversationId });
      
      const formattedMessages = [];
      conversation.dialogues.forEach(dialogue => {
        formattedMessages.push({ role: 'user', content: dialogue.prompt });
        if (dialogue.answer) {
          formattedMessages.push({ role: 'assistant', content: dialogue.answer });
        }
      });
      setMessages(formattedMessages);
      setCurrentView('chat');
    } catch (error) {
      console.error('Error loading conversation:', error);
    }
  };

  const sendMessage = () => {
    if (!inputMessage.trim() || !currentConversation) return;

    const userMessage = inputMessage;
    setInputMessage('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    // Send via WebSocket
    sendWebSocketMessage(currentConversation.id, userMessage);
  };

  const deleteConversation = async (conversationId) => {
    try {
      await axios.delete(`${API_BASE_URL}/conversation/${conversationId}`);
      await loadConversations();
      if (currentConversation?.id === conversationId) {
        setCurrentConversation(null);
        setMessages([]);
        setCurrentView('dashboard');
      }
    } catch (error) {
      console.error('Error deleting conversation:', error);
    }
  };

  // Login View
  if (currentView === 'login') {
    return (
      <div className="container">
        <div className="login-container">
          <h1>🤖 RAG Chat Demo</h1>
          <p>Select a user to continue</p>
          <div className="user-buttons">
            <button onClick={() => handleLogin('User1')} className="user-btn">
              👤 User 1
            </button>
            <button onClick={() => handleLogin('User2')} className="user-btn">
              👥 User 2
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Dashboard View
  if (currentView === 'dashboard') {
    return (
      <div className="container">
        <div className="sidebar">
          <div className="user-info">
            <span className="avatar">{user.avatar}</span>
            <div>
              <strong>{user.name}</strong>
              <small>@{user.username}</small>
            </div>
            <button onClick={handleLogout} className="logout-btn">🚪</button>
          </div>
          
          <button onClick={createNewConversation} className="new-chat-btn">
            ➕ New Conversation
          </button>
          
          <div className="conversations-list">
            <h3>Recent Conversations</h3>
            {conversations.length === 0 ? (
              <p className="no-conversations">No conversations yet</p>
            ) : (
              conversations.map(conv => (
                <div key={conv.id} className="conversation-item">
                  <button
                    onClick={() => loadConversation(conv.id)}
                    className="conversation-btn"
                  >
                    💬 {conv.title || conv.id.slice(0, 8)}...
                  </button>
                  <button
                    onClick={() => deleteConversation(conv.id)}
                    className="delete-conv-btn"
                  >
                    🗑️
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
        
        <div className="welcome-area">
          <h2>🎯 Welcome {user.name}!</h2>
          <p>Click "New Conversation" to start chatting with AI</p>
          <button onClick={createNewConversation} className="start-chat-btn">
            💬 Start New Chat
          </button>
        </div>
      </div>
    );
  }

  // Chat View
  return (
    <div className="container">
      <div className="sidebar">
        <div className="user-info">
          <span className="avatar">{user.avatar}</span>
          <div>
            <strong>{user.name}</strong>
            <small>@{user.username}</small>
          </div>
          <button onClick={() => setCurrentView('dashboard')} className="dashboard-btn">
            📋
          </button>
          <button onClick={handleLogout} className="logout-btn">🚪</button>
        </div>
        
        <button onClick={createNewConversation} className="new-chat-btn">
          ➕ New Conversation
        </button>
        
        <div className="conversations-list">
          <h3>Recent Conversations</h3>
          {conversations.length === 0 ? (
            <p className="no-conversations">No conversations yet</p>
          ) : (
            conversations.map(conv => (
              <div key={conv.id} className="conversation-item">
                <button
                  onClick={() => loadConversation(conv.id)}
                  className={`conversation-btn ${currentConversation?.id === conv.id ? 'active' : ''}`}
                >
                  💬 {conv.title || conv.id.slice(0, 8)}...
                </button>
                <button
                  onClick={() => deleteConversation(conv.id)}
                  className="delete-conv-btn"
                >
                  🗑️
                </button>
              </div>
            ))
          )}
        </div>
        
        <div className="connection-status">
          {wsConnected ? (
            <span className="status-connected">🟢 Connected</span>
          ) : (
            <span className="status-disconnected">🔴 Reconnecting...</span>
          )}
        </div>
      </div>
      
      <div className="chat-area">
        <div className="chat-header">
          <h3>💬 Chat</h3>
          <button onClick={createNewConversation} className="new-chat-header-btn">
            New
          </button>
        </div>
        
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-chat">
              <p>No messages yet. Start a conversation!</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? user.avatar : '🤖'}
                </div>
                <div className="message-content">
                  {msg.content}
                </div>
              </div>
            ))
          )}
          {loading && (
            <div className="message assistant">
              <div className="message-avatar">🤖</div>
              <div className="message-content thinking">
                🤔 Thinking...
              </div>
            </div>
          )}
        </div>
        
        <div className="input-area">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Type your message here..."
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={loading || !inputMessage.trim()}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;