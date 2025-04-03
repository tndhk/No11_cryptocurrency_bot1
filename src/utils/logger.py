import os
import sys
from loguru import logger

def setup_logger(log_level="INFO"):
    """ロギングの設定"""
    # 既存のハンドラをクリア
    logger.remove()
    
    # 標準出力へのロギング
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", log_level))
    
    # ファイルへのロギング
    logger.add("logs/trading_bot_{time}.log", rotation="1 day", retention="30 days")
    
    return logger

# デフォルトロガーのセットアップ
default_logger = setup_logger()