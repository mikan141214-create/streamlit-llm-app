import os
import streamlit as st

# --- dotenv が無くても落ちないフェイルセーフ ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass  # ローカルの .env が読めない場合でも、環境変数/Secrets で動く
# ---------- LangChain + OpenAI セットアップ ----------
# ライブラリ名は "langchain-openai"（ハイフン）ですが import は "langchain_openai"（アンダースコア）
from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore

# ---------- アプリのメタ情報 ----------
st.set_page_config(page_title="LLMアプリ（Lesson21）", page_icon="✨", layout="centered")

st.title("✨ Lesson21: Streamlit × LangChain LLMアプリ")
st.write(
    """
このアプリは、入力テキストを **LangChain** を介して **OpenAI** の LLM に渡し、  
選択した「専門家の振る舞い」で回答を生成します。

**使い方**  
1. 下のラジオボタンで「専門家の種類」を選びます  
2. テキスト欄に質問や文章を入力します  
3. 「実行」ボタンで回答を表示します

※ ローカル実行時は、プロジェクト直下の **.env** に `OPENAI_API_KEY=...` を記述してください。  
※ GitHub には `.env` を **絶対にアップロードしない** でください（`.gitignore` により除外）。
"""
)

st.divider()

# ---------- 専門家プロンプト定義 ----------
EXPERT_PROFILES = {
    "採用スペシャリスト": (
        "あなたは企業の人材採用におけるスペシャリストです。"
        "候補者体験、コンピテンシー、面接設計、スクリーニング、オンボーディング、"
        "タレントマネジメントとの連携を踏まえ、実務で使える提案を日本語で簡潔かつ具体的に出します。"
        "必要に応じて箇条書きと短い理由を併記してください。"
    ),
    "キャリアコーチ": (
        "あなたは経験豊富なキャリアコーチです。"
        "相談者の価値観・強み・コンディションに寄り添い、行動可能な次の一歩を日本語で提示します。"
        "選択肢は2〜3個、各選択肢に短い実行手順と注意点を添えてください。"
    ),
    "ヘルスケア管理栄養士": (
        "あなたはエビデンス志向の管理栄養士です。"
        "栄養学と行動変容の知見を用い、現実的な食事改善策を日本語で提案します。"
        "食材の代替案、外食時の工夫、継続のコツを具体的に挙げてください。"
    ),
}

def get_system_prompt(expert_key: str) -> str:
    """ラジオボタンの選択値から system メッセージを返す"""
    return EXPERT_PROFILES.get(
        expert_key,
        "あなたは有能なアシスタントです。ユーザーの目的達成を助けてください。"
    )

def answer_with_expert(input_text: str, expert_key: str) -> str:
    """
    条件の関数：
    - 引数: 「入力テキスト」「ラジオボタンの選択値」
    - 戻り値: LLMからの回答（文字列）
    """
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return (
            "【エラー】OPENAI_API_KEY が見つかりません。\n"
            "- ローカル: プロジェクト直下の .env に `OPENAI_API_KEY=...` を記述\n"
            "- Streamlit Cloud: App settings → Secrets に `OPENAI_API_KEY` を登録"
        )

    # ChatOpenAI の初期化（モデルは必要に応じて変更可）
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        api_key=api_key,
    )

    system_msg = get_system_prompt(expert_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "{input_text}"),
        ]
    )

    chain = prompt | llm
    resp = chain.invoke({"input_text": input_text})
    # langchain_openai.ChatOpenAI は AIMessage を返す想定
    return resp.content if hasattr(resp, "content") else str(resp)

# ---------- UI（フォーム） ----------
with st.form("llm_form"):
    expert = st.radio(
        "専門家の種類を選んでください：",
        list(EXPERT_PROFILES.keys()),
        index=0,
        horizontal=False,
    )
    user_input = st.text_area(
        "質問や文章を入力してください：",
        placeholder="例）未経験でデータ分析職に挑戦するには何から始めるべき？",
        height=140,
    )
    submitted = st.form_submit_button("実行")

if submitted:
    if not user_input.strip():
        st.error("入力テキストを入力してください。")
    else:
        with st.spinner("LLMに問い合わせ中…"):
            output = answer_with_expert(user_input, expert)
        st.divider()
        st.write("#### 回答")
        st.write(output)
