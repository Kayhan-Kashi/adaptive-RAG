# elearning-service/src/dependencies.py
from __future__ import annotations
import os
import logging

from injector import Binder, Module, SingletonScope, Injector
from common.kafka.producer import KafkaProducer, get_producer
from services.conversation_service import ConversationService
from services.document_service import DocumentService

logger = logging.getLogger(__name__)


def get_kafka_producer() -> KafkaProducer:
    """Factory function for Kafka producer"""
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    logger.info(f"Creating Kafka producer with bootstrap: {bootstrap_servers}")
    return get_producer(bootstrap_servers=bootstrap_servers)


class DependencyInjection(Module):
    """Dependency injection configuration"""
    
    def configure(self, binder: Binder):
        logger.info("Configuring DI bindings...")
        binder.bind(KafkaProducer, to=get_kafka_producer, scope=SingletonScope)
        binder.bind(ConversationService, scope=SingletonScope)
        binder.bind(DocumentService, scope=SingletonScope)  # This must be here
        logger.info("DI bindings configured")


# Create injector instance
injector = Injector([DependencyInjection()])