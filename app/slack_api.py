import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
from dotenv import load_dotenv
from app.llm import LLMManager
from app.rag import RAGEngine

load_dotenv()

# Slack Bot Token, App Tokenは環境変数から取得
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "xoxb-your-bot-token")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "xapp-your-app-token")

app = App(token=SLACK_BOT_TOKEN)

# LLM/RAGエンジン初期化（プロセス起動時に1度だけ）
llm_manager = LLMManager()
rag_engine = RAGEngine()
rag_engine.initialize()

@app.event("app_mention")
def handle(event, say):
    text = event["text"]
    user = event.get("user")
    # メンション部分を除去（<@Uxxxx> など）
    import re
    clean_text = re.sub(r"<@[^>]+>\s*", "", text).strip()
    if not clean_text:
        say("質問内容が見つかりませんでした。@bot名 の後に質問を入力してください。")
        return
    try:
        llm = llm_manager.get_llm()
        result = rag_engine.query(clean_text, llm)
        answer = result["answer"]
        sources = result.get("source_documents", [])
        # ソース情報を整形
        if sources:
            srcs = "\n".join(f"- {s['metadata'].get('source', '')}" for s in sources)
            answer += f"\n\n参考: \n{srcs}"
        say(f"<@{user}> {answer}")
    except Exception as e:
        say(f"エラー: {e}")


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()


