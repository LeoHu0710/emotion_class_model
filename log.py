import io
import sys
import logging

# 將控制台輸出編碼設為 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 創建一個 logger 物件
train_and_test_logger = logging.getLogger("train")
train_and_test_logger .setLevel(logging.INFO)

api_logger = logging.getLogger("api")
api_logger .setLevel(logging.INFO)

# 創建一個檔案處理器，將日誌訊息寫入檔案
train_file_handler = logging.FileHandler('train_and_test.log', encoding='utf-8')
train_file_handler.setLevel(logging.INFO)

api_file_handler = logging.FileHandler('api.log', encoding='utf-8')
api_file_handler.setLevel(logging.INFO)


# 創建一個格式化器，定義日誌訊息的格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
train_file_handler.setFormatter(formatter)
api_file_handler.setFormatter(formatter)

# 將檔案處理器添加到 logger 物件
train_and_test_logger.addHandler(train_file_handler)
api_logger.addHandler(api_file_handler)