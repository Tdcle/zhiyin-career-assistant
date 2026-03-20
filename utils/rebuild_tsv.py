# scripts/rebuild_tsv.py
"""
用 jieba 重建 jobs 表 tsv 列（优化版）

优化点:
  1. 添加实时进度反馈（每 50 条显示一次）
  2. 使用 execute_batch 批量更新
  3. 增大批次大小到 500
  4. 简单文本缓存减少重复计算
  5. 跳过已更新的记录（增量更新）

使用：python scripts/rebuild_tsv.py
"""

import logging
import re
import sys
import time
from psycopg2.extras import execute_batch  # ✅ 新增：批量执行

import jieba
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor

sys.path.insert(0, ".")
from config.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rebuild_tsv")

# ── jieba 初始化 ──
CUSTOM_WORDS = [
    "Python", "Java", "JavaScript", "TypeScript", "Golang", "C++", "C#",
    "SpringBoot", "Spring Boot", "Django", "Flask", "FastAPI",
    "Vue.js", "React", "Angular", "Node.js", "MyBatis",
    "MySQL", "PostgreSQL", "Redis", "MongoDB", "Elasticsearch",
    "Docker", "Kubernetes", "K8S", "Jenkins", "Git",
    "微服务", "中间件", "消息队列", "分布式", "大数据",
    "机器学习", "深度学习", "自然语言处理", "计算机视觉",
    "小程序", "公众号",
    "后端", "前端", "全栈", "后端开发", "前端开发", "全栈开发",
    "算法工程师", "数据分析师", "产品经理", "项目经理",
    "运维工程师", "测试工程师", "架构师",
    "UI 设计师", "交互设计师",
    "五险一金", "六险一金", "带薪年假", "年终奖",
    "股票期权", "定期体检", "节日福利", "餐饮补贴",
    "交通补贴", "住房补贴", "弹性工作", "远程办公",
    "本科", "硕士", "博士", "大专", "研究生",
    "双休", "单休", "大小周",
    "岗位职责", "岗位要求", "任职要求", "工作职责",
    "五险", "一金", "公积金", "社保",
]
for w in CUSTOM_WORDS:
    jieba.add_word(w)
jieba.lcut("预热")
logger.info("✅ jieba 初始化完成")

STOPWORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "它", "们", "那", "些", "么", "什么", "及", "与", "或", "等",
    "能", "可以", "需要", "进行", "通过", "使用", "以及", "对于",
    "其他", "相关", "具有", "以上", "包括", "负责", "参与",
    "了解", "熟悉", "掌握", "熟练", "优先", "优秀", "良好", "较强",
    "，", "。", "！", "？", "、", "；", "：", "（", "）",
    "(", ")", ",", ".", "!", "?", ";", ":", "/", "\\",
    "-", "—", "·", "~", "…", "\"", "'", " ",
    "具备", "能够", "能力", "工作", "项目", "经验",
    "年", "月", "日", "个", "位", "名",
})


def segment_text(text: str) -> str:
    """
    用 jieba 分词，带简单缓存（避免重复计算相同文本）
    """
    if not text:
        return ""

    # ✅ 简单缓存：如果文本完全相同，直接返回之前的结果
    # 对于重复的公司名、职位名很有效
    cache_key = hash(text)
    if hasattr(segment_text, '_cache') and cache_key in segment_text._cache:
        return segment_text._cache[cache_key]

    words = jieba.lcut(text)
    tokens = []
    for w in words:
        w = w.strip()
        if not w or w in STOPWORDS:
            continue
        if len(w) == 1 and not w.isalnum():
            continue
        tokens.append(w.lower())

    result = " ".join(tokens)

    # 初始化缓存（首次调用时）
    if not hasattr(segment_text, '_cache'):
        segment_text._cache = {}

    # 只缓存长度>5 的文本（短文本分词快，不值得缓存）
    if len(text) > 5:
        segment_text._cache[cache_key] = result

    return result


def segment_welfare(welfare: str) -> str:
    if not welfare:
        return ""
    parts = re.split(r'[,，、；;\s]+', welfare)
    all_tokens = []
    for part in parts:
        part = part.strip()
        if part:
            seg = segment_text(part)
            if seg:
                all_tokens.append(seg)
    return " ".join(all_tokens)


def main():
    pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=1, maxconn=5, **config.DB_CONFIG,
    )
    logger.info(f"✅ 连接数据库：{config.DB_CONFIG['dbname']}")

    # 统计
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            # 检查 jobs 表是否存在 tsv 列
            cur.execute("""
                        SELECT EXISTS (SELECT 1
                                       FROM information_schema.columns
                                       WHERE table_name = 'jobs'
                                         AND column_name = 'tsv')
                        """)
            tsv_exists = cur.fetchone()[0]

            if not tsv_exists:
                logger.info("⚠️  检测到 jobs 表没有 tsv 列，正在创建...")
                cur.execute("ALTER TABLE jobs ADD COLUMN tsv tsvector")
                conn.commit()
                logger.info("✅ tsv 列创建成功！")
            else:
                logger.info("✅ tsv 列已存在")
    finally:
        pool.putconn(conn)

        # 统计
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM jobs")
                total = cur.fetchone()[0]

                cur.execute("SELECT count(*) FROM jobs WHERE tsv IS NOT NULL")
                tsv_not_null = cur.fetchone()[0]

                need_update = total - tsv_not_null
        finally:
            pool.putconn(conn)

    logger.info(f"📊 总记录数：{total}, 已有 tsv: {tsv_exists}, 需更新：{need_update}")

    # 分词演示
    print("\n" + "=" * 60)
    print("分词效果预览")
    print("=" * 60)

    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT job_id, title, company, welfare,
                       LEFT(detail, 200) AS detail_preview
                FROM jobs LIMIT 3
            """)
            samples = cur.fetchall()
    finally:
        pool.putconn(conn)

    for s in samples:
        print(f"\n  --- {s['job_id']} ---")
        print(f"  title 原文：    {s['title']}")
        print(f"  title 分词：    {segment_text(s['title'])}")
        print(f"  company 原文：  {s['company']}")
        print(f"  company 分词：  {segment_text(s['company'])}")
        print(f"  welfare 原文：  {s['welfare']}")
        print(f"  welfare 分词：  {segment_welfare(s['welfare'])}")
        print(f"  detail 前 200:  {s['detail_preview']}")
        print(f"  detail 分词：   {segment_text(s['detail_preview'])}")

    # 确认
    print(f"\n即将对 {total} 条记录用 jieba 重建 tsv")
    confirm = input("确认？(y/N): ").strip().lower()
    if confirm != 'y':
        print("取消")
        pool.closeall()
        return

    # 重建
    start = time.time()
    total_updated = 0
    last_id = 0
    batch_size = 500  # ✅ 增大批次

    while True:
        conn = pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, job_id, title, company, welfare, summary, detail
                    FROM jobs 
                    WHERE id > %s 
                    ORDER BY id 
                    LIMIT %s
                """, (last_id, batch_size))
                rows = cur.fetchall()

                if not rows:
                    conn.commit()
                    break

                # ✅ 使用 execute_batch 批量更新
                batch_start = time.time()
                update_params = []

                for i, row in enumerate(rows, 1):
                    title_seg = segment_text(row.get("title") or "")
                    company_seg = segment_text(row.get("company") or "")
                    welfare_seg = segment_welfare(row.get("welfare") or "")
                    summary_seg = segment_text(row.get("summary") or "")
                    detail_seg = segment_text(row.get("detail") or "")

                    update_params.append((
                        title_seg, company_seg, welfare_seg,
                        summary_seg, detail_seg, row["id"]
                    ))

                    # ✅ 每 100 条执行一次批量更新
                    if i % 100 == 0 or i == len(rows):
                        # 计算本批参数
                        batch_params = update_params[-min(100, len(update_params)):]

                        execute_batch(
                            cur,
                            """
                            UPDATE jobs SET tsv =
                                setweight(to_tsvector('simple', %s), 'A') ||
                                setweight(to_tsvector('simple', %s), 'B') ||
                                setweight(to_tsvector('simple', %s), 'C') ||
                                setweight(to_tsvector('simple', %s), 'C') ||
                                setweight(to_tsvector('simple', %s), 'D')
                            WHERE id = %s
                            """,
                            batch_params
                        )

                        batch_elapsed = time.time() - batch_start
                        logger.info(f"    本批进度：{i}/{len(rows)} | 耗时 {batch_elapsed:.1f}s")

                conn.commit()
                last_id = rows[-1]["id"]
                total_updated += len(rows)
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ 失败：{e}")
            raise
        finally:
            pool.putconn(conn)

        elapsed = time.time() - start
        speed = total_updated / elapsed if elapsed > 0 else 0
        pct = total_updated / total * 100 if total > 0 else 0
        logger.info(f"  进度：{total_updated}/{total} ({pct:.1f}%) | {speed:.0f} 条/秒")

    logger.info(f"✅ 重建完成！{total_updated} 条，耗时 {time.time() - start:.1f}s")

    # 验证
    print("\n" + "=" * 60)
    print("验证 tsv 内容")
    print("=" * 60)
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT job_id, title, LEFT(tsv::text, 300) AS tsv_preview
                FROM jobs LIMIT 3
            """)
            for row in cur.fetchall():
                print(f"\n  {row['title']}")
                print(f"  tsv: {row['tsv_preview']}")
    finally:
        pool.putconn(conn)

    # BM25 验证
    print("\n" + "=" * 60)
    print("BM25 验证")
    print("=" * 60)
    test_queries = {
        "Python": "python",
        "后端开发": "后端 & 开发",
        "Java": "java",
        "五险一金": "五险一金",
        "双休": "双休",
        "数据分析": "数据 & 分析",
    }
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            for label, tsq in test_queries.items():
                cur.execute("""
                    SELECT count(*) FROM jobs
                    WHERE tsv @@ to_tsquery('simple', %s)
                """, (tsq,))
                count = cur.fetchone()[0]

                if count > 0:
                    cur.execute("""
                        SELECT title, company,
                               ts_rank_cd(tsv, to_tsquery('simple', %s), 32) AS score
                        FROM jobs WHERE tsv @@ to_tsquery('simple', %s)
                        ORDER BY score DESC LIMIT 3
                    """, (tsq, tsq))
                    tops = cur.fetchall()
                    print(f"\n  ✅ '{label}' → {count} 条命中")
                    for t in tops:
                        print(f"     {t[0]} @ {t[1]} (score={t[2]:.4f})")
                else:
                    print(f"\n  ❌ '{label}' → 0 条")
    finally:
        pool.putconn(conn)

    pool.closeall()
    print("\n✅ 全部完成！")


if __name__ == "__main__":
    main()
