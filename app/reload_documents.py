import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import RAGEngine

rag_engine = RAGEngine()

try:
    rag_engine.initialize(force_reload=True)
    print("Documents reloaded successfully")
except Exception as e:
    raise Exception(f"Failed to reload documents: {str(e)}")
