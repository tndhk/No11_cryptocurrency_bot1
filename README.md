# 高度なトレーディングボット（リファクタリング版）

Dockerを利用した自動取引ボットの高度な開発環境です。複合シグナル戦略、詳細な価格シミュレーション、取引コスト考慮など多くの機能を実装しています。

## 主な機能

- **複合シグナルロジック**: 複数の指標を重み付けして統合
- **詳細な価格シミュレーション**: ローソク足内の価格変動を再現
- **取引コスト考慮**: 手数料とスリッページを現実的にモデル化
- **高度なバックテスト**: リアルな市場状況を反映
- **最適化エンジン**: 戦略パラメータの並列自動最適化
- **効率的なデータ管理**: 履歴データのキャッシュ
- **詳細な分析ツール**: 取引結果の高度な視覚化

## リファクタリングの改善点

- **モジュール化**: 単一の大きなクラスを複数の特化したモジュールに分割
- **責任の分離**: 各コンポーネントが特定の機能に特化
- **並列処理**: 最適化処理を複数CPUコアで実行し高速化
- **保守性向上**: コードの整理と再利用性の向上
- **テスト容易性**: 各モジュールが独立して機能するためテスト容易

## プロジェクト構造

```
trading-bot/
├── config/               # 設定関連
│   ├── __init__.py
│   └── settings.py       # 設定管理
├── src/                  # ソースコード
│   ├── __init__.py
│   ├── data/             # データ取得・管理
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── indicators/       # テクニカル指標計算
│   │   ├── __init__.py
│   │   └── technical.py
│   ├── signals/          # シグナル生成
│   │   ├── __init__.py
│   │   └── signal_generator.py
│   ├── risk/             # リスク管理
│   │   ├── __init__.py
│   │   └── risk_manager.py
│   ├── execution/        # 注文実行
│   │   ├── __init__.py
│   │   └── order_executor.py
│   ├── backtest/         # バックテスト
│   │   ├── __init__.py
│   │   └── backtester.py
│   ├── optimization/     # パラメータ最適化
│   │   ├── __init__.py
│   │   └── optimizer.py
│   └── utils/            # ユーティリティ
│       ├── __init__.py
│       ├── logger.py
│       └── metrics.py
├── data/                 # データ保存
├── logs/                 # ログファイル
├── results/              # バックテスト結果
├── cache/                # データキャッシュ
├── models/               # 学習済みモデル
├── Dockerfile            # Docker イメージ定義
├── docker-compose.yml    # サービス構成
├── requirements.txt      # Python 依存関係
└── trading_bot.py        # メインアプリケーション
```

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

最適なパラメータの自動探索（並列処理対応）:

```bash
docker-compose run trading-bot python trading_bot.py --mode optimize
```

または専用サービスを使用:

```bash
docker-compose up optimization
```

```
# 重要なパラメータのみを最適化（最速）
docker-compose run -e OPTIMIZATION_MODE=advanced -e OPTIMIZE_IMPORTANT_ONLY=true trading-bot python trading_bot.py --mode optimize
```
```
# 段階的なベイズ最適化
docker-compose run -e OPTIMIZATION_MODE=advanced -e OPTIMIZE_IMPORTANT_ONLY=false trading-bot python trading_bot.py --mode optimize
```
```
# 従来のグリッドサーチ
docker-compose run -e OPTIMIZATION_MODE=grid trading-bot python trading_bot.py --mode optimize
```
```
# パラメータ重要度分析のみ実行
docker-compose run trading-bot python trading_bot.py --mode analyze
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

### 並行処理設定

最適化の並列処理ワーカー数の調整:

```
OPTIMIZATION_WORKERS=0  # 0 = 自動（CPUコア数 - 1）
```

## 最適化結果の解釈

### 最適化視覚化

- **パラメータ重要度**: 各パラメータの利益への影響グラフ
- **相関分析**: パラメータ間の相互作用

## リスク警告

このトレーディングボットは教育および研究目的で提供されています。実際のマーケットで使用する前に十分なテストとリスク評価を行ってください。暗号通貨や金融市場での取引には重大な財務リスクが伴います。

## ライセンス

[MIT License](LICENSE)