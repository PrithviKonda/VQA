# src/knowledge/connectors/base_connector.py
"""
Base connector for external knowledge sources.
"""
class BaseConnector:
    def retrieve(self, query):
        raise NotImplementedError
