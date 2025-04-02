# 高度なトレーディングボット

Dockerを利用した自動取引ボットの高度な開発環境です。複合シグナル戦略、詳細な価格シミュレーション、取引コスト考慮など多くの機能を実装しています。

## 主な機能

- **複合シグナルロジック**: 複数の指標を重み付けして統合
- **詳細な価格シミュレーション**: ローソク足内の価格変動を再現
- **取引コスト考慮**: 手数料とスリッページを現実的にモデル化
- **高度なバックテスト**: リアルな市場状況を反映
- **最適化エンジン**: 戦略パラメータの自動最適化
- **効率的なデータ管理**: 履歴データのキャッシュ
- **詳細な分析ツール**: 取引結果の高度な視覚化

## セットアップと使用方法

### 前提条件

- Docker と Docker Compose がインストールされていること
- Binance アカウントと API キー（テスト環境または本番環境）

### 初期設定

1. リポジトリのクローン:

```bash
git clone <repository-url>
cd trading-bot
```

2. 環境変数ファイルの設定:

`.env` ファイルを編集して、API キーと戦略パラメータを設定します。

3. Docker コンテナのビルドと起動:

```bash
docker-compose build
docker-compose up -d
```

### 使用方法

#### バックテストモード

過去データを使用した戦略のテスト:

```bash
docker-compose run trading-bot python trading_bot.py --mode backtest
```

#### パラメータ最適化

最適なパラメータの自動探索:

```bash
docker-compose run trading-bot python trading_bot.py --mode optimize
```

または専用サービスを使用:

```bash
docker-compose up optimization
```

#### ライブトレードモード

実際の市場での取引（注意してください！）:

```bash
docker-compose run trading-bot python trading_bot.py --mode live
```

#### Jupyter Notebook

高度な分析とカスタマイズ:

```bash
docker-compose up jupyter
```

ブラウザで `http://localhost:8888` にアクセスします。

## 戦略のカスタマイズ

### 複合シグナル調整

`.env` ファイルで各指標の重みを調整できます:

```
WEIGHT_MA=0.3    # 移動平均
WEIGHT_RSI=0.2   # RSI
WEIGHT_MACD=0.2  # MACD
WEIGHT_BB=0.2    # ボリンジャーバンド
WEIGHT_BREAKOUT=0.1  # ブレイクアウト
```

### 価格シミュレーション

価格変動シミュレーションのパラメータを調整:

```
USE_PRICE_SIMULATION=true
PRICE_SIMULATION_STEPS=100
```

### 取引コスト

実際の市場状況に合わせて調整:

```
MAKER_FEE=0.0010
TAKER_FEE=0.0010
SLIPPAGE_MEAN=0.0005
SLIPPAGE_STD=0.0003
```

## プロジェクト構造

```
.
├── Dockerfile              # Docker イメージ定義
├── docker-compose.yml      # サービス構成
├── requirements.txt        # Python 依存関係
├── .env                    # 環境変数（非公開）
├── trading_bot.py          # メインアプリケーション
├── data/                   # データ保存ディレクトリ
├── logs/                   # ログファイル
├── results/                # バックテスト結果
├── cache/                  # データキャッシュ
├── models/                 # 学習済みモデル
└── notebooks/              # Jupyter ノートブック
```

## 分析結果の解釈

### バックテスト視覚化

- **価格チャート**: 取引ポイントとテクニカル指標
- **残高推移**: ドローダウン分析付き
- **RSI/MACD**: シグナル発生ポイントの視覚化
- **取引分析**: 理由別パフォーマンス、月別収益など

### 最適化結果

- **パラメータ重要度**: 各パラメータの利益への影響
- **相関分析**: パラメータ間の相互作用

## リスク警告

このトレーディングボットは教育および研究目的で提供されています。実際のマーケットで使用する前に十分なテストとリスク評価を行ってください。暗号通貨や金融市場での取引には重大な財務リスクが伴います。

## ライセンス

[MIT License](LICENSE)