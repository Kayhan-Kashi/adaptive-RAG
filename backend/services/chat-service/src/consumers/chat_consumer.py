import asyncio
import json
import logging
from typing import Optional
from datetime import datetime
from confluent_kafka import Consumer, KafkaError
from api.websocket.manager import connection_manager

logger = logging.getLogger(__name__)


class ChatConsumer:
    """Dedicated Kafka consumer for chat chunk streaming events"""
    
    def __init__(self, bootstrap_servers: str, group_id: str, topic: str):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topic = topic
        self.consumer: Optional[Consumer] = None
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.message_count = 0
        self.chunk_count = 0
    
    def start(self):
        """Start the consumer"""
        logger.info("=" * 60)
        logger.info("🔄 STARTING CHAT CONSUMER")
        logger.info("=" * 60)
        logger.info(f"📡 Bootstrap: {self.bootstrap_servers}")
        logger.info(f"📡 Group ID: {self.group_id}")
        logger.info(f"📡 Topic: {self.topic}")
        
        conf = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 5000,
            'session.timeout.ms': 30000,
            'max.poll.interval.ms': 300000,
        }
        
        try:
            self.consumer = Consumer(conf)
            self.consumer.subscribe([self.topic])
            self.running = True
            
            logger.info("✅ Consumer subscribed successfully")
            
            # List available topics
            try:
                metadata = self.consumer.list_topics(timeout=10)
                available_topics = list(metadata.topics.keys())
                logger.info(f"📋 Available topics: {available_topics}")
                if self.topic in available_topics:
                    logger.info(f"✅ Topic '{self.topic}' exists")
                    logger.info(f"   Partitions: {len(metadata.topics[self.topic].partitions)}")
                else:
                    logger.warning(f"⚠️ Topic '{self.topic}' does NOT exist!")
            except Exception as e:
                logger.warning(f"⚠️ Could not list topics: {e}")
            
            logger.info("=" * 60)
            logger.info("✅ ChatConsumer started successfully")
            logger.info("=" * 60)
                
        except Exception as e:
            logger.error(f"❌ Failed to start consumer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def stop(self):
        """Stop the consumer"""
        self.running = False
        if self.consumer:
            self.consumer.close()
            logger.info("ChatConsumer stopped")
    
    async def consume(self):
        """Background task to consume messages"""
        if not self.consumer:
            raise RuntimeError("ChatConsumer not started")
        
        loop = asyncio.get_event_loop()
        logger.info("🔄 Consumer loop started, waiting for messages...")
        logger.info(f"⏳ Polling topic: {self.topic}")
        
        while self.running:
            try:
                msg = await loop.run_in_executor(None, self.consumer.poll, 1.0)
                
                if msg is None:
                    await asyncio.sleep(0.1)
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"❌ Consumer error: {msg.error()}")
                        continue
                
                # ✅ Message received!
                self.message_count += 1
                
                logger.info("=" * 60)
                logger.info(f"📩 RECEIVED MESSAGE #{self.message_count}")
                logger.info(f"   Offset: {msg.offset()}")
                logger.info(f"   Partition: {msg.partition()}")
                logger.info(f"   Timestamp: {datetime.utcnow().isoformat()}")
                logger.info("=" * 60)
                
                try:
                    message = json.loads(msg.value().decode('utf-8'))
                    
                    # ✅ Log full message (truncated for readability)
                    event_type = message.get("event_type")
                    conversation_id = message.get("conversation_id", "unknown")[:8]
                    dialogue_id = message.get("dialogue_id", "unknown")[:8]
                    
                    logger.info(f"📦 Message details:")
                    logger.info(f"   Event Type: {event_type}")
                    logger.info(f"   Conversation: {conversation_id}...")
                    logger.info(f"   Dialogue: {dialogue_id}...")
                    
                    if event_type == "prompt_answer_chunk_streamed":
                        chunk = message.get("chunk", "")
                        chunk_index = message.get("chunk_index", 0)
                        is_last = message.get("is_last", False)
                        
                        self.chunk_count += 1
                        logger.info(f"   🧩 CHUNK #{self.chunk_count}:")
                        logger.info(f"      Index: {chunk_index}")
                        logger.info(f"      Length: {len(chunk)} characters")
                        logger.info(f"      Is Last: {is_last}")
                        logger.info(f"      Content: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
                    
                    await self._process_message(message)
                    logger.info(f"✅ Message {self.message_count} processed successfully")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to decode message: {e}")
                    logger.error(f"   Raw message: {msg.value()[:200]}")
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
            except Exception as e:
                logger.error(f"❌ Error in consume loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)
        
        logger.info("🔄 Consumer loop stopped")
        logger.info(f"📊 Total messages consumed: {self.message_count}")
        logger.info(f"📊 Total chunks processed: {self.chunk_count}")
    
    async def _process_message(self, message: dict):
        """Process incoming Kafka message"""
        try:
            event_type = message.get("event_type")
            
            logger.info(f"🔄 Processing event type: {event_type}")
            
            if event_type == "prompt_answer_chunk_streamed":
                await self._handle_chunk_event(message)
            else:
                logger.warning(f"⚠️ Unknown event type: {event_type}")
                logger.warning(f"   Full message: {json.dumps(message, indent=2)[:500]}")
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _handle_chunk_event(self, message: dict):
        """Handle prompt_answer_chunk_streamed event"""
        conversation_id = message.get("conversation_id")
        dialogue_id = message.get("dialogue_id")
        chunk = message.get("chunk", "")
        chunk_index = message.get("chunk_index", 0)
        is_last = message.get("is_last", False)
        
        logger.info("=" * 50)
        logger.info(f"🧩 HANDLING CHUNK EVENT")
        logger.info(f"   Conversation ID: {conversation_id}")
        logger.info(f"   Dialogue ID: {dialogue_id}")
        logger.info(f"   Chunk Index: {chunk_index}")
        logger.info(f"   Chunk Length: {len(chunk)} chars")
        logger.info(f"   Is Last: {is_last}")
        logger.info(f"   Chunk Preview: {chunk[:80]}{'...' if len(chunk) > 80 else ''}")
        logger.info("=" * 50)
        
        # ✅ Check WebSocket routing
        logger.info(f"🔍 Checking WebSocket routing:")
        logger.info(f"   Active connections: {list(connection_manager.active_connections.keys())}")
        logger.info(f"   Conversation routing: {connection_manager.conversation_routing}")
        
        user_id = connection_manager.conversation_routing.get(conversation_id)
        if user_id:
            logger.info(f"   ✅ Found user_id: {user_id}")
            if user_id in connection_manager.active_connections:
                logger.info(f"   ✅ User {user_id} is connected")
            else:
                logger.warning(f"   ⚠️ User {user_id} is NOT connected")
        else:
            logger.warning(f"   ⚠️ No user_id found for conversation {conversation_id}")
        
        # Send chunk via WebSocket
        logger.info(f"📤 Sending chunk to WebSocket...")
        success = await connection_manager.send_chunk(
            conversation_id=conversation_id,
            chunk=chunk,
            chunk_index=chunk_index,
            is_last=is_last
        )
        
        if success:
            if is_last:
                logger.info(f"✅✅✅ FINAL CHUNK SENT SUCCESSFULLY for dialogue {dialogue_id[:8]}...")
                logger.info(f"   Total chunks in stream: {chunk_index + 1}")
            else:
                logger.info(f"✅ Chunk {chunk_index} sent via WebSocket successfully")
        else:
            logger.warning(f"⚠️ WebSocket send failed for conversation {conversation_id[:8]}...")
            logger.warning(f"   Check if user is connected and conversation is registered")


# ============================================
# Helper to create and start the consumer
# ============================================

_chat_consumer: Optional[ChatConsumer] = None


def start_chat_consumer(
    bootstrap_servers: str,
    group_id: str,
    topic: str = "prompt-answer-chunk-streamed"
) -> ChatConsumer:
    """Create and start the chat consumer"""
    global _chat_consumer
    
    _chat_consumer = ChatConsumer(bootstrap_servers, group_id, topic)
    _chat_consumer.start()
    
    # Start consume loop
    loop = asyncio.get_event_loop()
    loop.create_task(_chat_consumer.consume())
    
    return _chat_consumer


def stop_chat_consumer():
    """Stop the chat consumer"""
    global _chat_consumer
    if _chat_consumer:
        _chat_consumer.stop()
        _chat_consumer = None