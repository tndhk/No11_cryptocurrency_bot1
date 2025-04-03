#!/bin/bash
# 最適化実行スクリプト

# ログファイル名のタイムスタンプを生成
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="logs/optimization_$timestamp.log"

# ヘルプメッセージを表示する関数
show_help() {
    echo "使用方法: $0 [オプション]"
    echo
    echo "オプション:"
    echo "  -m, --mode MODE        最適化モード (advanced, grid)"
    echo "  -i, --important        重要なパラメータのみを最適化 (true, false)"
    echo "  -n, --iterations N     イテレーション数"
    echo "  -w, --workers N        ワーカー数 (CPUコア数)"
    echo "  -a, --algorithm ALG    アルゴリズム (bayesian, forest, gbrt)"
    echo "  -h, --help             このヘルプを表示"
    echo
    echo "例:"
    echo "  $0 -m advanced -i true -n 50 -a bayesian"
    echo "  $0 -m grid"
    echo "  $0 --mode advanced --important false --iterations 100"
}

# デフォルト値
mode="advanced"
important="true"
iterations="50"
workers="0"
algorithm="bayesian"

# 引数のパース
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--mode) mode="$2"; shift ;;
        -i|--important) important="$2"; shift ;;
        -n|--iterations) iterations="$2"; shift ;;
        -w|--workers) workers="$2"; shift ;;
        -a|--algorithm) algorithm="$2"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "不明なオプション: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo "最適化設定:"
echo "モード: $mode"
echo "重要パラメータのみ: $important"
echo "イテレーション数: $iterations"
echo "ワーカー数: $workers"
echo "アルゴリズム: $algorithm"
echo "ログファイル: $logfile"
echo "=========================================="

# コマンドを構築
cmd="docker-compose run \
    -e OPTIMIZATION_MODE=$mode \
    -e OPTIMIZE_IMPORTANT_ONLY=$important \
    -e OPTIMIZATION_ITERATIONS=$iterations \
    -e OPTIMIZATION_WORKERS=$workers \
    -e OPTIMIZATION_METHOD=$algorithm \
    trading-bot python trading_bot.py --mode optimize"

# コマンドをログに記録
echo "実行コマンド: $cmd" | tee -a "$logfile"
echo "開始時刻: $(date)" | tee -a "$logfile"
echo "=========================================="

# コマンドを実行し、結果をログファイルに保存
eval "$cmd" | tee -a "$logfile"

echo "=========================================="
echo "終了時刻: $(date)" | tee -a "$logfile"
echo "結果とログはresultsディレクトリとlogs/$logfile に保存されました"