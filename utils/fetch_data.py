import random
import time
from urllib.parse import quote

from DrissionPage import ChromiumPage
from langchain_core.messages import HumanMessage

from config.config import config
from utils.database import DatabaseManager
from utils.logger import get_logger

logger = get_logger("fetch_data")
db = DatabaseManager()
summary_llm = config.create_tongyi(config.CHAT_MODELS.data_summary)


def generate_summary(detail_text: str) -> str:
    """生成与当前检索系统一致的职位摘要。"""
    if not detail_text:
        return ""
    if len(detail_text.strip()) < 50:
        return detail_text.strip()[:150]

    short_text = detail_text[:3000]
    prompt = f"""
请阅读以下职位描述，提炼核心信息。
要求：
1. 提炼核心技术栈、硬性要求、岗位职责。
2. 去除废话和重复描述。
3. 控制在 100 字以内。
4. 直接输出摘要正文，不要输出 JSON。

职位描述：
{short_text}
"""
    try:
        response = summary_llm.invoke([HumanMessage(content=prompt)])
        summary = (response.content or "").strip()
        return summary or detail_text.strip()[:150]
    except Exception as exc:
        logger.error("summary generation failed: %s", exc, exc_info=True)
        return detail_text.strip()[:150]


def build_job_data_from_list_item(job: dict) -> dict:
    labels = job.get("jobLabels") or []
    return {
        "job_id": job["encryptJobId"],
        "title": job.get("jobName", ""),
        "salary": job.get("salaryDesc", ""),
        "company": job.get("brandName", ""),
        "industry": job.get("brandIndustry", ""),
        "city": job.get("cityName", ""),
        "district": job.get("areaDistrict", ""),
        "experience": labels[0] if len(labels) > 0 else "",
        "degree": labels[1] if len(labels) > 1 else "",
        "welfare": ",".join(job.get("welfareList", [])),
        "detail_url": f"https://www.zhipin.com/job_detail/{job['encryptJobId']}.html",
        "detail": "",
        "summary": "",
    }


def persist_job_record(job_data: dict) -> str:
    detail_text = (job_data.get("detail") or "").strip()
    if not detail_text:
        logger.warning("skip job without detail: %s", job_data.get("job_id"))
        return "missing_detail"

    summary = generate_summary(detail_text)
    job_data["summary"] = summary

    embedding_text = db.build_job_embedding_text(job_data, summary=summary)
    try:
        vector = db.embed_model.embed_query(embedding_text)
    except Exception as exc:
        logger.error("embedding generation failed for job=%s: %s", job_data.get("job_id"), exc, exc_info=True)
        return "embedding_failed"

    if not vector or len(vector) != config.VECTOR_DIM:
        logger.error("invalid embedding vector for job=%s", job_data.get("job_id"))
        return "embedding_failed"

    return db.save_job_with_analysis(job_data, summary=summary, vector=vector)


def crawl_job_detail(page: ChromiumPage, detail_url: str) -> str:
    tab = page.new_tab(detail_url)
    try:
        if not tab.ele(".job-sec-text", timeout=6):
            return ""
        tab.scroll.down(random.randint(200, 500))
        detail_ele = tab.ele(".job-sec-text")
        return detail_ele.text if detail_ele else ""
    finally:
        try:
            tab.close()
        except Exception:
            pass


def run_crawler():
    print("\n=== 启动抓取并直接入库模式 ===")

    keyword = input("请输入搜索关键词（例如 java）: ").strip() or "java"
    target_count_str = input("请输入计划抓取数量（例如 100）: ").strip()
    target_count = int(target_count_str) if target_count_str.isdigit() else 100

    print(f"\n>>> 任务配置: 搜索 [{keyword}] | 目标数量 [{target_count}]")

    page = ChromiumPage()
    page.listen.start("wapi/zpgeek/search/joblist.json")

    safe_keyword = quote(keyword)
    # target_url = f"https://www.zhipin.com/web/geek/job?query={safe_keyword}&city=101020100"
    target_url = f"https://www.zhipin.com/web/geek/jobs?city=101210100&jobType=1902&query={safe_keyword}"
    page.get(target_url)
    print(">>> 浏览器已打开，开始抓取并直接完成摘要、向量、TSV 入库...")

    current_count = 0

    while current_count < target_count:
        res = page.listen.wait(timeout=5)
        if not res:
            print("⌛ 等待数据中，自动下滑加载更多...")
            page.scroll.to_bottom()
            time.sleep(random.uniform(1.5, 2.5))
            continue

        body = res.response.body
        if not isinstance(body, dict) or "zpData" not in body:
            continue

        job_list = body["zpData"].get("jobList", [])
        if not job_list:
            print("⚠️ 未获取到职位列表，尝试继续滚动。")
            page.scroll.to_bottom()
            time.sleep(2)
            continue

        print(f"\n--- 本页捕获 {len(job_list)} 条数据（当前进度: {current_count}/{target_count}）---")

        for job in job_list:
            if current_count >= target_count:
                break

            try:
                job_data = build_job_data_from_list_item(job)
                current_count += 1
                print(f"[{current_count}/{target_count}] 处理: {job_data['title']}", end=" ")

                detail = crawl_job_detail(page, job_data["detail_url"])
                if not detail:
                    print("⚠️ 详情页加载失败")
                    time.sleep(random.uniform(1.0, 2.0))
                    continue

                job_data["detail"] = detail
                status = persist_job_record(job_data)

                if status == "inserted":
                    print("✅ 新增并完成分析")
                elif status == "updated":
                    print("♻️ 更新并完成分析")
                elif status == "embedding_failed":
                    print("❌ 向量生成失败")
                else:
                    print("❌ 入库失败")

                time.sleep(random.uniform(1.5, 3.0))
            except Exception as exc:
                logger.error("crawler item failed: %s", exc, exc_info=True)
                print(f"❌ 单条错误: {exc}")

        if current_count < target_count:
            print(">>> 本批数据处理完毕，自动下滑加载更多...")
            page.scroll.to_bottom()
            time.sleep(random.uniform(2.0, 4.0))

    print(f"\n🎉 抓取结束，共处理 {current_count} 条职位。")


def run_processor():
    print("\n=== 启动数据补处理模式 ===")
    print("将扫描 summary 为空或 embedding 为空的职位，并按当前项目规则补全。")

    while True:
        jobs = db.fetch_jobs_pending_analysis(limit=10)
        if not jobs:
            print("🎉 所有待补全职位已处理完成。")
            break

        print(f"\n⚙️ 本批次处理 {len(jobs)} 条职位...")

        for job in jobs:
            job_data = dict(job)
            title = job_data.get("title", "")
            print(f"处理中: {title[:20]}...", end="")
            start_t = time.time()

            status = persist_job_record(job_data)
            elapsed = time.time() - start_t

            if status in {"inserted", "updated"}:
                print(f" ✅ 完成 ({status}, {elapsed:.2f}s)")
            else:
                print(f" ❌ 失败 ({status})")

        print("--- 休息 1 秒 ---")
        time.sleep(1)


if __name__ == "__main__":
    print("================ Job RAG System ================")
    print("1. 抓取并直接入库（摘要 + 向量 + TSV）")
    print("2. 补处理历史数据（summary / embedding 缺失）")
    print("================================================")

    choice = input("请输入功能序号 (1/2): ").strip()
    if choice == "1":
        run_crawler()
    elif choice == "2":
        run_processor()
    else:
        print("无效输入")
