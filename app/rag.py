"""
RAG (Retrieval-Augmented Generation) エンジン
"""
from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGEngine:
    """RAG機能を提供するクラス"""
    
    def __init__(
        self,
        data_dir: str = "data",
        persist_directory: str = "chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        RAGエンジンの初期化
        
        Args:
            data_dir: ドキュメントが保存されているディレクトリ
            persist_directory: ベクトルストアの永続化ディレクトリ
            chunk_size: テキスト分割のチャンクサイズ
            chunk_overlap: チャンク間のオーバーラップ
        """
        self.data_dir = Path(data_dir)
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Embeddingsモデルの初期化
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.vectorstore: Optional[Chroma] = None
        self.rag_chain = None
        
    def load_documents(self):
        """データディレクトリからドキュメントを読み込む"""
        documents = []
        
        # テキストファイルの読み込み
        if self.data_dir.exists():
            txt_files = list(self.data_dir.glob("**/*.txt"))
            for txt_file in txt_files:
                try:
                    loader = TextLoader(str(txt_file), encoding="utf-8")
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {txt_file}: {e}")
                    
            # PDFファイルの読み込み (オプション)
            pdf_files = list(self.data_dir.glob("**/*.pdf"))
            for pdf_file in pdf_files:
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {pdf_file}: {e}")
        
        return documents
    
    def create_vectorstore(self, documents: list):
        """ドキュメントからベクトルストアを作成"""
        # テキストの分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        
        # ベクトルストアの作成
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        
        return len(texts)
    
    def load_vectorstore(self):
        """既存のベクトルストアを読み込む"""
        if Path(self.persist_directory).exists():
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            return True
        return False
    
    def initialize(self, force_reload: bool = False):
        """
        RAGエンジンを初期化
        
        Args:
            force_reload: Trueの場合、既存のベクトルストアを無視して再作成
        """
        if not force_reload and self.load_vectorstore():
            print("Loaded existing vector store")
            return
            
        print("Creating new vector store...")
        documents = self.load_documents()
        
        if not documents:
            print("Warning: No documents found in data directory")
            return
            
        num_chunks = self.create_vectorstore(documents)
        print(f"Created vector store with {num_chunks} chunks from {len(documents)} documents")
    
    def create_rag_chain(self, llm: BaseChatModel):
        """
        RAGチェーンを作成
        
        Args:
            llm: 使用するLLMモデル
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        
        # プロンプトテンプレートの作成
        template = """以下のコンテキストを使用して質問に答えてください。
コンテキストに答えがない場合は、「提供された情報では答えられません」と答えてください。

コンテキスト:
{context}

質問: {question}

回答:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Retrieverの作成
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # 上位3件の関連ドキュメントを取得
        )
        
        # RAGチェーンの作成
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return retriever
        
    def query(self, question: str, llm: BaseChatModel) -> dict:
        """
        質問に対して回答を生成
        
        Args:
            question: ユーザーの質問
            llm: 使用するLLMモデル
            
        Returns:
            回答とソースドキュメントを含む辞書
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        
        # RAGチェーンの作成
        retriever = self.create_rag_chain(llm)
        
        # ソースドキュメントの取得
        source_docs = retriever.invoke(question)
        
        # 質問の実行
        answer = self.rag_chain.invoke(question)
        
        return {
            "answer": answer,
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in source_docs
            ]
        }
