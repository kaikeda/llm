# LLM Chat API - RAGシステム

Gemini と Claude を切り替え可能な RAG (Retrieval-Augmented Generation) チャットシステムです。FastAPI で構築され、LangChain を使用しています。

## 機能

- 🤖 **LLMモデル切り替え**: Gemini と Claude を簡単に切り替え
- 📚 **RAG機能**: ドキュメントベースの質問応答
- 🔄 **リアルタイムドキュメント更新**: ドキュメントの動的な再読み込み
- 🚀 **高速API**: FastAPIによる高速なレスポンス
- 📊 **ベクトルデータベース**: ChromaDBによる効率的な検索

## プロジェクト構成

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPIアプリケーション
│   ├── llm.py           # LLMマネージャー
│   └── rag.py           # RAGエンジン
├── data/                # ドキュメント保存ディレクトリ
├── chroma_db/           # ベクトルデータベース (自動生成)
├── .env                 # 環境変数 (要作成)
├── .env.example         # 環境変数のサンプル
├── requirements.txt     # 依存パッケージ
└── README.md
```

## セットアップ

### 1. 環境変数の設定

`.env.example` をコピーして `.env` ファイルを作成し、APIキーを設定してください。

```bash
cp .env.example .env
```

`.env` ファイルを編集:

```env
# Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Anthropic API Key
ANTHROPIC_API_KEY=your_claude_api_key_here

# デフォルトLLMモデル (gemini or claude)
DEFAULT_LLM=gemini

# Gemini モデル名
GEMINI_MODEL=gemini-2.0-flash-exp

# Claude モデル名
CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

### 2. APIキーの取得

- **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey) で取得
- **Claude**: [Anthropic Console](https://console.anthropic.com/) で取得

### 3. ドキュメントの配置

`data/` ディレクトリに質問応答の元となるドキュメント（`.txt` または `.pdf`）を配置してください。

```bash
# 例: サンプルドキュメントの作成
echo "これはサンプルドキュメントです。" > data/sample.txt
```

## 起動方法

### 開発サーバーの起動

```bash
# 仮想環境のアクティベート（既にアクティブな場合はスキップ）
source .venv/bin/activate

# FastAPIサーバーの起動
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

サーバーが起動したら、以下のURLでアクセスできます:

- **API**: http://localhost:8000
- **対話的ドキュメント (Swagger UI)**: http://localhost:8000/docs
- **代替ドキュメント (ReDoc)**: http://localhost:8000/redoc

## API使用例

### 1. ヘルスチェック

```bash
curl http://localhost:8000/health
```

### 2. 利用可能なモデル一覧

```bash
curl http://localhost:8000/models
```

### 3. RAGを使用したチャット

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ドキュメントの内容について教えてください",
    "model": "gemini",
    "use_rag": true
  }'
```

レスポンス例:
```json
{
  "answer": "ドキュメントによると...",
  "model_used": "gemini",
  "sources": [
    {
      "content": "関連するドキュメントの内容",
      "metadata": {"source": "data/sample.txt"}
    }
  ]
}
```

### 4. シンプルなチャット (RAGなし)

```bash
curl -X POST http://localhost:8000/simple-chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "こんにちは！",
    "model": "claude"
  }'
```

### 5. モデルの切り替え

```bash
# Geminiを使用
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "質問内容", "model": "gemini"}'

# Claudeを使用
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "質問内容", "model": "claude"}'
```

### 6. ドキュメントの再読み込み

新しいドキュメントを `data/` ディレクトリに追加した後:

```bash
curl -X POST http://localhost:8000/reload-documents
```

## Pythonからの使用例

```python
import requests

# APIのベースURL
BASE_URL = "http://localhost:8000"

# RAGを使用したチャット
def chat_with_rag(question: str, model: str = "gemini"):
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "question": question,
            "model": model,
            "use_rag": True
        }
    )
    return response.json()

# 使用例
result = chat_with_rag("ドキュメントについて教えて", model="gemini")
print(f"Answer: {result['answer']}")
print(f"Model: {result['model_used']}")
```

## トラブルシューティング

### エラー: "GOOGLE_API_KEY is not set"

- `.env` ファイルが正しく作成されているか確認
- `.env` ファイルにAPIキーが設定されているか確認

### エラー: "No documents found in data directory"

- `data/` ディレクトリにドキュメントファイルを配置してください
- サポートされている形式: `.txt`, `.pdf`

### RAGが正しく動作しない

1. ドキュメントを再読み込み: `POST /reload-documents`
2. `chroma_db/` ディレクトリを削除して再起動

## 開発

### テストの実行

```bash
# 必要に応じてpytestをインストール
pip install pytest pytest-asyncio httpx

# テストの実行
pytest
```

### コードフォーマット

```bash
# blackをインストール
pip install black

# フォーマット
black app/
```

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！

## 参考資料

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Gemini API Documentation](https://ai.google.dev/)
- [Claude API Documentation](https://docs.anthropic.com/)