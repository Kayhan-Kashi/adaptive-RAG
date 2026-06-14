# elearning-service/src/api/routes/websocket_routes.py
import logging
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi_injector import Injected
from sqlmodel import Session

from api.websocket.manager import connection_manager
from database.sqlite_session import get_session
from services.conversation_service import ConversationService

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    user_id: str,
    db: Session = Depends(get_session),
    service: ConversationService = Injected(ConversationService)
):
    """WebSocket endpoint for real-time communication"""
    
    logger.info(f"🔌 WebSocket connection attempt from user: {user_id}")
    
    try:
        # Accept the connection
        await websocket.accept()
        logger.info(f"✅ WebSocket accepted for user: {user_id}")
        
        # Register the connection
        await connection_manager.connect(websocket, user_id)
        logger.info(f"📝 User {user_id} registered in connection manager")
        
        # Keep the connection alive and handle messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()
                msg_type = data.get("type")
                
                logger.info(f"📨 Received message from user {user_id}: {msg_type}")
                logger.debug(f"   Message data: {data}")
                
                # Handle ping/pong
                if msg_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    logger.debug(f"🏓 Sent pong to user {user_id}")
                
                # Handle chat messages
                elif msg_type == "chat":
                    conversation_id = data.get("conversation_id")
                    prompt = data.get("prompt")
                    
                    logger.info(f"💬 Chat message from user {user_id} in conversation {conversation_id}")
                    logger.info(f"   Prompt: {prompt[:100]}...")
                    
                    # Register conversation for WebSocket routing
                    connection_manager.register_conversation(conversation_id, user_id)
                    
                    # Create dialogue - this will automatically publish Kafka event
                    dialogue_id = await service.create_dialogue(
                        session=db,
                        conversation_id=conversation_id,
                        prompt=prompt,
                        answer=None
                    )
                    
                    logger.info(f"📝 Dialogue created: {dialogue_id}")
                    logger.info(f"📤 Kafka event published by ConversationService")
                    
                    # Send acknowledgment
                    await websocket.send_json({
                        "type": "ack",
                        "conversation_id": conversation_id,
                        "dialogue_id": str(dialogue_id),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    logger.debug(f"✅ Sent acknowledgment to user {user_id}")
                    
                    # Note: The answer will come from the ChatConsumer 
                    # (which listens to prompt-answer-completed topic)
                    # and will be sent via connection_manager.send_answer()
                
                # Handle unknown message types
                else:
                    logger.warning(f"⚠️ Unknown message type from user {user_id}: {msg_type}")
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Unknown message type: {msg_type}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
            except WebSocketDisconnect:
                logger.info(f"❌ User {user_id} disconnected")
                break
                
            except Exception as e:
                logger.error(f"❌ Error processing message from user {user_id}: {e}")
                logger.exception("Stack trace:")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except:
                    pass
                
    except Exception as e:
        logger.error(f"❌ WebSocket error for user {user_id}: {e}")
        logger.exception("Stack trace:")
        await websocket.close()