# Streamlit LLM App

LangChainとOpenAIを使用した、専門家チャットボットアプリケーションです。

## 機能

- 3種類の専門家プロフィール（採用スペシャリスト、キャリアコーチ、ヘルスケア管理栄養士）
- OpenAI GPT-4o-miniを使用した自然言語処理
- LangChainによる効率的なプロンプトエンジニアリング

## 🚀 Streamlit Cloudへのデプロイ

### 前提条件

- GitHubアカウント
- OpenAI APIキー（[OpenAI](https://platform.openai.com/)で取得）

### デプロイ手順

1. **このリポジトリをフォークまたはクローン**

2. **Streamlit Cloudにアクセス**
   - [share.streamlit.io](https://share.streamlit.io/)にアクセス
   - GitHubアカウントでサインイン

3. **新しいアプリをデプロイ**
   - 「New app」をクリック
   - リポジトリ、ブランチ、メインファイル（`app.py`）を選択
   - 「Deploy!」をクリック

4. **シークレット（APIキー）の設定**
   - デプロイ後、App settings → Secrets に移動
   - 以下の形式でOpenAI APIキーを追加：
     ```toml
     OPENAI_API_KEY = "sk-your-actual-api-key-here"
     ```
   - 「Save」をクリック

5. **アプリが自動的に再起動し、利用可能になります**

## 🔧 ローカル開発

### セットアップ

1. **リポジトリをクローン**
   ```bash
   git clone <repository-url>
   cd streamlit-llm-app
   ```

2. **仮想環境を作成（推奨）**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **依存関係をインストール**
   ```bash
   pip install -r requirements.txt
   ```

4. **環境変数の設定**

   プロジェクトルートに `.env` ファイルを作成：
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

   または、`.streamlit/secrets.toml` を作成：
   ```toml
   OPENAI_API_KEY = "sk-your-actual-api-key-here"
   ```

5. **アプリを起動**
   ```bash
   streamlit run app.py
   ```

6. ブラウザで `http://localhost:8501` を開く

## 📁 プロジェクト構成

```
streamlit-llm-app/
├── app.py                          # メインアプリケーション
├── requirements.txt                # Python依存関係
├── README.md                       # このファイル
├── .gitignore                      # Git除外設定
├── .streamlit/
│   ├── config.toml                 # Streamlit設定
│   └── secrets.toml.example        # シークレット設定例
└── .env                            # 環境変数（ローカル開発用、Git除外）
```

## 🔐 セキュリティ注意事項

- `.env` ファイルや `.streamlit/secrets.toml` は **絶対にGitにコミットしない**でください
- `.gitignore` で適切に除外されていることを確認してください
- APIキーは定期的にローテーションすることを推奨します

## 📝 使用技術

- [Streamlit](https://streamlit.io/) - Webアプリフレームワーク
- [LangChain](https://python.langchain.com/) - LLMアプリケーションフレームワーク
- [OpenAI](https://openai.com/) - 言語モデルAPI

## ライセンス

MIT License
