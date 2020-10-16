import logging
import os
from setting import info

file_path = os.path.join("./log", info)
logger = logging.getLogger("log")
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("{}.txt".format(file_path))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)

# -----------------------------------
logger_count = logging.getLogger("count_log")
logger_count.setLevel(level=logging.INFO)
handler = logging.FileHandler("{}_count.txt".format(file_path))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger_count.addHandler(handler)
