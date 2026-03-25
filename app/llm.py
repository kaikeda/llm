"""
LLMマネージャー - Gemini と Claude の切り替え
"""
import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel


class LLMManager:
    """LLMモデルの切り替えを管理するクラス"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        self.default_llm = os.getenv("DEFAULT_LLM", "gemini")
        
    def get_llm(self, model_type: Optional[str] = None) -> BaseChatModel:
        """
        指定されたLLMモデルを取得
        
        Args:
            model_type: "gemini" または "claude"。Noneの場合はデフォルト
            
        Returns:
            LangChainのチャットモデルインスタンス
        """
        if model_type is None:
            model_type = self.default_llm
            
        if model_type.lower() == "gemini":
            return self._get_gemini()
        elif model_type.lower() == "claude":
            return self._get_claude()
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'gemini' or 'claude'")
    
    def _get_gemini(self) -> ChatGoogleGenerativeAI:
        """Gemini モデルを取得"""
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY is not set in environment variables")
        
        return ChatGoogleGenerativeAI(
            model=self.gemini_model,
            google_api_key=self.gemini_api_key,
            temperature=0.7,
        )
    
    def _get_claude(self) -> ChatAnthropic:
        """Claude モデルを取得"""
        if not self.claude_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment variables")
        
        return ChatAnthropic(
            model=self.claude_model,
            anthropic_api_key=self.claude_api_key,
            temperature=0.7,
        )
    
    def get_available_models(self) -> list[str]:
        """
        利用可能なモデルのリストを返す
        
        Returns:
            利用可能なモデル名のリスト
        """
        available = []
        if self.gemini_api_key:
            available.append("gemini")
        if self.claude_api_key:
            available.append("claude")
        return available
