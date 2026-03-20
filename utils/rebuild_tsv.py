"""
兼容维护脚本。

当前项目的职位入库和分析补处理已经会自动维护 `summary`、`embedding` 和 `tsv`，
通常不再需要单独运行本脚本。

仅当你怀疑旧数据的 `tsv` 列需要整库回填时，再手动执行：

    python utils/rebuild_tsv.py
"""

from utils.database import DatabaseManager


def main():
    db = DatabaseManager()
    total = db.backfill_tsv(batch_size=200)
    print(f"tsv 回填完成，共处理 {total} 条职位。")


if __name__ == "__main__":
    main()
