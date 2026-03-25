"""
FastAPI チャットアプリケーション - LLM問い合わせチャット
Gemini / Claude 切り替え可能なRAGシステム
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from app.llm import LLMManager
from app.rag import RAGEngine

# 環境変数の読み込み
load_dotenv()

# FastAPIアプリケーションの初期化
app = FastAPI(
    title="LLM Chat API",
    description="Gemini/Claude切り替え可能なRAGチャットシステム",
    version="1.0.0"
)

# LLMマネージャーとRAGエンジンの初期化
llm_manager = LLMManager()
rag_engine = RAGEngine()

# 起動時にRAGエンジンを初期化
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の処理"""
    try:
        rag_engine.initialize()
        print("RAG engine initialized successfully")
    except Exception as e:
        print(f"Warning: RAG engine initialization failed: {e}")


# リクエスト/レスポンスモデル
class ChatRequest(BaseModel):
    """チャットリクエストのモデル"""
    question: str
    model: Optional[str] = None  # "gemini" または "claude"
    use_rag: bool = True  # RAGを使用するかどうか


class ChatResponse(BaseModel):
    """チャットレスポンスのモデル"""
    answer: str
    model_used: str
    sources: Optional[list] = None


class SimpleRequest(BaseModel):
    """シンプルなチャットリクエスト"""
    message: str
    model: Optional[str] = None


# エンドポイント
@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "LLM Chat API",
        "available_models": llm_manager.get_available_models(),
        "default_model": llm_manager.default_llm
    }


@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "rag_initialized": rag_engine.vectorstore is not None
    }


@app.get("/models")
async def get_models():
    """利用可能なモデル一覧を取得"""
    return {
        "available_models": llm_manager.get_available_models(),
        "default_model": llm_manager.default_llm
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAGを使用したチャットエンドポイント
    
    Args:
        request: チャットリクエスト
        
    Returns:
        回答とソース情報
    """
    try:
        # LLMモデルの取得
        llm = llm_manager.get_llm(request.model)
        model_used = request.model or llm_manager.default_llm
        
        if request.use_rag:
            # RAGを使用した回答生成
            result = rag_engine.query(request.question, llm)
            return ChatResponse(
                answer=result["answer"],
                model_used=model_used,
                sources=result["source_documents"]
            )
        else:
            # LLMのみで回答生成
            response = llm.invoke(request.question)
            return ChatResponse(
                answer=response.content,
                model_used=model_used,
                sources=None
            )
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/simple-chat")
async def simple_chat(request: SimpleRequest):
    """
    シンプルなチャットエンドポイント (RAGなし)
    
    Args:
        request: チャットリクエスト
        
    Returns:
        回答
    """
    try:
        # LLMモデルの取得
        llm = llm_manager.get_llm(request.model)
        model_used = request.model or llm_manager.default_llm
        
        # LLMで回答生成
        response = llm.invoke(request.message)
        
        return {
            "answer": response.content,
            "model_used": model_used
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/reload-documents")
async def reload_documents():
    """
    ドキュメントを再読み込みしてベクトルストアを更新
    """
    try:
        rag_engine.initialize(force_reload=True)
        return {"message": "Documents reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload documents: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
