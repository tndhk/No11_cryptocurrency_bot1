# トレーディングボット開発用Dockerfile
FROM python:3.10-slim

# タイムゾーンを設定（日本時間）
ENV TZ=Asia/Tokyo

# 作業ディレクトリを設定
WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 特定のバージョンの最適化ライブラリをインストール（互換性のため）
RUN pip install --no-cache-dir scikit-optimize==0.9.0 scikit-learn==1.2.2 numpy==1.24.3

# ソースコードをコピー
COPY . .

# 環境変数を設定 (本番環境では.envファイルかシークレット管理を推奨)
ENV PYTHONUNBUFFERED=1

# コンテナ起動時に実行するコマンド（デフォルトはバックテスト）
CMD ["python", "trading_bot.py", "--mode", "backtest"]