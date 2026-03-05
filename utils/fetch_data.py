import time
import random
import dashscope
from urllib.parse import quote  # 新增：用于URL编码
from http import HTTPStatus
from DrissionPage import ChromiumPage
from utils.database import DatabaseManager
from config.config import config

# ================= 配置区域 =================
SUMMARY_MODEL = "qwen-turbo"


# ================= 工具函数 (保持不变) =================
def generate_summary(detail_text):
    """
    调用通义千问 API 生成摘要
    """
    if not detail_text or len(detail_text) < 50:
        return detail_text

    short_text = detail_text[:3000]

    prompt = f"""
    请阅读以下职位描述，提炼核心信息。
    要求：
    1. 提炼核心技术栈(python/前端/java)、硬性要求（学历/年限）、岗位职责。
    2. 去除废话。
    3. 控制在 100 字以内。
    4. 直接输出摘要。

    职位描述：
    {short_text}
    """

    try:
        dashscope.api_key = config.DASHSCOPE_API_KEY
        response = dashscope.Generation.call(
            model=SUMMARY_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.text
        return detail_text[:100]
    except Exception as e:
        print(f"⚠️ 摘要生成异常: {e}")
        return detail_text[:100]


# ================= 模块一：极速爬虫 (修改版) =================
def run_crawler():
    print("\n=== 启动极速爬虫模式 ===")

    # 1. 获取用户输入
    keyword = input("请输入搜索关键词 (例如 java): ").strip()
    if not keyword: keyword = "java"  # 默认值

    target_count_str = input("请输入计划爬取的数量 (例如 100): ").strip()
    target_count = int(target_count_str) if target_count_str.isdigit() else 100

    print(f"\n>>> 任务配置: 搜索 [{keyword}] | 目标数量 [{target_count}]")

    db = DatabaseManager()
    dp = ChromiumPage()

    # 监听数据包
    dp.listen.start('wapi/zpgeek/search/joblist.json')

    # 构造URL (处理中文编码)
    safe_keyword = quote(keyword)
    target_url = f'https://www.zhipin.com/web/geek/job?query={safe_keyword}&city=101010100'

    dp.get(target_url)
    print(f">>> 浏览器已打开，开始工作...")

    current_count = 0

    # 循环直到达到目标数量
    while current_count < target_count:
        # 设置等待时间，如果5秒没刷出新数据包，说明需要滚动或者到底了
        res = dp.listen.wait(timeout=5)

        # === 情况A: 还没有捕获到数据包 (超时) ===
        if not res:
            print("⌛ 等待数据中，尝试自动下滑加载更多...")
            dp.scroll.to_bottom()  # 滚到底部触发加载
            time.sleep(random.uniform(1.5, 2.5))  # 给页面一点反应时间
            continue

        # === 情况B: 捕获到数据包 ===
        if not res.response.body: continue

        json_data = res.response.body
        if not isinstance(json_data, dict) or 'zpData' not in json_data:
            continue

        job_list = json_data['zpData'].get('jobList', [])
        if not job_list:
            print("⚠️ 未获取到职位列表，可能已到达底部或触发验证码。")
            # 可以在这里加一个检查验证码的逻辑，或者多试几次滚动
            dp.scroll.to_bottom()
            time.sleep(2)
            continue

        print(f"\n--- 本页捕获 {len(job_list)} 条数据 (当前进度: {current_count}/{target_count}) ---")

        for i, job in enumerate(job_list):
            # 如果已经达到目标数量，停止处理
            if current_count >= target_count:
                print(f"\n🎉 已达到目标数量 {target_count}，停止抓取。")
                break

            try:
                # 1. 基础数据提取
                job_id = job['encryptJobId']

                # 检查是否已存在 (可选优化：防止重复打开详情页)
                # if db.check_exists(job_id): continue

                job_data = {
                    'job_id': job_id,
                    'title': job['jobName'],
                    'salary': job['salaryDesc'],
                    'company': job['brandName'],
                    'industry': job['brandIndustry'],
                    'city': job['cityName'],
                    'district': job['areaDistrict'],
                    'experience': job['jobLabels'][0] if job.get('jobLabels') else '',
                    'degree': job['jobLabels'][1] if len(job.get('jobLabels', [])) > 1 else '',
                    'welfare': ','.join(job.get('welfareList', [])),
                    'detail_url': f"https://www.zhipin.com/job_detail/{job_id}.html",
                    'detail': '',
                    'summary': ''
                }

                current_count += 1
                print(f"[{current_count}/{target_count}] 抓取: {job_data['title']}", end=" ")

                # 2. 详情页抓取
                tab = dp.new_tab(job_data['detail_url'])

                # 等待详情元素加载
                if tab.ele('.job-sec-text', timeout=6):
                    # 详情页内小幅滚动，模拟真人阅读
                    tab.scroll.down(random.randint(200, 500))
                    job_data['detail'] = tab.ele('.job-sec-text').text

                    if db.insert_job(job_data):
                        print("✅")
                    else:
                        print("♻️ 重复")
                else:
                    print("⚠️ 详情页加载超时")

                tab.close()
                # 详情页抓取间隔
                time.sleep(random.uniform(1.5, 3.0))

            except Exception as e:
                print(f"❌ 单条错误: {e}")
                try:
                    tab.close()
                except:
                    pass

        # === 批次处理完毕，准备加载下一页 ===
        if current_count < target_count:
            print(">>> 本批数据处理完毕，自动下滑加载更多...")
            dp.scroll.to_bottom()
            # 随机休眠，模拟翻页阅读停顿
            time.sleep(random.uniform(2.0, 4.0))

    print(f"\n🏁 爬虫任务结束，共抓取 {current_count} 条数据。")


# ================= 模块二：数据工厂 (保持不变) =================
def run_processor():
    print("\n=== 启动数据处理模式 (Qwen摘要 + Ollama向量) ===")
    print("正在扫描未处理的数据 (Embedding 为空的记录)...")

    db = DatabaseManager()

    while True:
        jobs = db.fetch_jobs_without_embedding(limit=10)
        if not jobs:
            print("🎉 所有数据处理完毕！")
            break

        print(f"\n⚡ 本批次处理 {len(jobs)} 条数据...")

        for job in jobs:
            job_id, title, company, salary, welfare, detail, city, district, experience, degree = job

            print(f"正在处理: {title[:10]}...", end="")
            start_t = time.time()

            summary = generate_summary(detail)

            text_to_embed = (
                f"职位: {title} | "
                f"地点: {city} {district} | "
                f"公司: {company} | "
                f"薪资: {salary} | "
                f"经验要求: {experience} | "
                f"学历要求: {degree} | "
                f"福利: {welfare} | "
                f"介绍: {summary}"
            )

            vector = db.embed_model.embed_query(text_to_embed)

            if not vector:
                print(" ❌ 向量生成失败")
                continue

            if db.update_job_analysis(job_id, summary, vector):
                cost = time.time() - start_t
                print(f" ✅ 完成 (耗时 {cost:.2f}s)")
            else:
                print(" ❌ 数据库更新失败")

        print("--- 休息 1 秒 ---")
        time.sleep(1)

    db.close()


# ================= 主程序入口 =================
if __name__ == "__main__":
    print("================ Job RAG System ================")
    print("1. 极速爬虫 (自动翻页 + 指定数量)")
    print("2. 数据工厂 (后台生成摘要 + 向量化)")
    print("================================================")

    choice = input("请输入功能序号 (1/2): ")

    if choice == '1':
        run_crawler()
    elif choice == '2':
        run_processor()
    else:
        print("无效输入")