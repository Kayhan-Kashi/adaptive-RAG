# api/routes/websocket.py

import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi_injector import Injected
from sqlmodel import Session

from api.websocket.manager import connection_manager
from database.sqlite_session import get_session
from services.conversation_service import ConversationService

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
        await websocket.accept()
        logger.info(f"✅ WebSocket accepted for user: {user_id}")
        
        await connection_manager.connect(websocket, user_id)
        logger.info(f"📝 User {user_id} registered in connection manager")
        
        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get("type")
                
                logger.info(f"📨 Received message from user {user_id}: {msg_type}")
                logger.debug(f"   Message data: {data}")
                
                if msg_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    logger.debug(f"🏓 Sent pong to user {user_id}")
                
                # ✅ Handle register_conversation message
                elif msg_type == "register_conversation":
                    conversation_id = data.get("conversation_id")
                    if conversation_id:
                        connection_manager.register_conversation(conversation_id, user_id)
                        await websocket.send_json({
                            "type": "ack",
                            "message": "Conversation registered successfully",
                            "conversation_id": conversation_id,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        logger.info(f"✅ Conversation {conversation_id[:8]}... registered for user {user_id}")
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "error": "Missing conversation_id",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        logger.warning(f"⚠️ register_conversation missing conversation_id from user {user_id}")
                
                elif msg_type == "chat":
                    conversation_id = data.get("conversation_id")
                    prompt = data.get("prompt")
                    file_ids: Optional[List[str]] = data.get("file_ids", [])
                    
                    # ============================================================
                    # Extract RAG Configuration from WebSocket message
                    # ============================================================
                    retrieval_k = data.get("retrieval_k", 20)
                    similarity_threshold = data.get("similarity_threshold", 0.5)
                    min_docs_required = data.get("min_docs_required", 3)
                    top_k = data.get("top_k", 5)
                    use_hyde = data.get("use_hyde", True)
                    sparse_ratio = data.get("sparse_ratio", 0.2)
                    retrieval_total_k = data.get("retrieval_total_k", 20)
                    use_reranker = data.get("use_reranker", True)
                    use_mmr = data.get("use_mmr", True)
                    mmr_fetch_k = data.get("mmr_fetch_k", 200)
                    mmr_lambda_mult = data.get("mmr_lambda_mult", 0.8)
                    
                    logger.info(f"💬 Chat message from user {user_id} in conversation {conversation_id}")
                    logger.info(f"   Prompt: {prompt[:100]}...")
                    logger.info(f"   File IDs: {file_ids}")
                    logger.info(f"   ⚙️ RAG Config: retrieval_k={retrieval_k}, threshold={similarity_threshold}, top_k={top_k}")
                    logger.info(f"   HyDE={use_hyde}, MMR={use_mmr}, lambda={mmr_lambda_mult}")
                    
                    connection_manager.register_conversation(conversation_id, user_id)
                    
                    dialogue_id = await service.create_dialogue(
                        session=db,
                        conversation_id=conversation_id,
                        prompt=prompt,
                        answer=None,
                        file_ids=file_ids,
                        # ============================================================
                        # Pass all RAG configuration parameters
                        # ============================================================
                        retrieval_k=retrieval_k,
                        similarity_threshold=similarity_threshold,
                        min_docs_required=min_docs_required,
                        top_k=top_k,
                        use_hyde=use_hyde,
                        sparse_ratio=sparse_ratio,
                        retrieval_total_k=retrieval_total_k,
                        use_reranker=use_reranker,
                        use_mmr=use_mmr,
                        mmr_fetch_k=mmr_fetch_k,
                        mmr_lambda_mult=mmr_lambda_mult,
                    )
                    
                    logger.info(f"📝 Dialogue created: {dialogue_id}")
                    logger.info(f"   Associated file IDs: {file_ids}")
                    logger.info("📤 Kafka event published by ConversationService")
                    
                    await websocket.send_json({
                        "type": "ack",
                        "conversation_id": conversation_id,
                        "dialogue_id": str(dialogue_id),
                        "file_ids": file_ids,
                        "config": {
                            "retrieval_k": retrieval_k,
                            "similarity_threshold": similarity_threshold,
                            "top_k": top_k,
                            "use_hyde": use_hyde,
                            "use_mmr": use_mmr,
                            "mmr_lambda_mult": mmr_lambda_mult,
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    logger.debug(f"✅ Sent acknowledgment to user {user_id}")
                
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