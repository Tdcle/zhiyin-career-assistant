import time
import random
import dashscope
from http import HTTPStatus
from DrissionPage import ChromiumPage
from tools.database import DatabaseManager
from config.config import config

# ================= 配置区域 =================
SUMMARY_MODEL = "qwen-turbo"


# ================= 工具函数 =================
def generate_summary(detail_text):
    """
    调用通义千问 API 生成摘要
    """
    if not detail_text or len(detail_text) < 50:
        return detail_text

    # 截取防止 token 溢出
    short_text = detail_text[:3000]

    prompt = f"""
    请阅读以下职位描述，提炼核心信息。
    要求：
    1. 提炼核心技术栈、硬性要求（学历/年限）、岗位职责。
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
        return detail_text[:100]  # 降级
    except Exception as e:
        print(f"⚠️ 摘要生成异常: {e}")
        return detail_text[:100]


# ================= 模块一：极速爬虫 (只管抓) =================
def run_crawler():
    print("\n=== 启动极速爬虫模式 (只抓取，不处理) ===")
    db = DatabaseManager()
    dp = ChromiumPage()

    dp.listen.start('wapi/zpgeek/search/joblist.json')

    target_url = 'https://www.zhipin.com/web/geek/job?query=java开发&city=101010100'
    dp.get(target_url)
    print(f">>> 浏览器已打开: {target_url}")
    print(">>> 请手动翻页...")

    while True:
        res = dp.listen.wait()
        if not res or not res.response.body: continue

        json_data = res.response.body
        if not isinstance(json_data, dict) or 'zpData' not in json_data: continue

        job_list = json_data['zpData'].get('jobList', [])
        if not job_list: continue

        print(f"\n--- 捕获到 {len(job_list)} 条数据 ---")

        for i, job in enumerate(job_list):
            try:
                # 1. 基础数据
                job_id = job['encryptJobId']

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
                    'summary': ''  # 爬虫阶段留空
                }

                print(f"[{i + 1}/{len(job_list)}] 抓取: {job_data['title']}", end=" ")

                # 2. 详情页抓取
                tab = dp.new_tab(job_data['detail_url'])
                if tab.ele('.job-sec-text', timeout=8):
                    tab.scroll.down(random.randint(200, 500))
                    job_data['detail'] = tab.ele('.job-sec-text').text

                    # 3. 直接入库 (不调 AI，速度快)
                    if db.insert_job(job_data):
                        print("✅ 已存原始数据")
                    else:
                        print("♻️ 重复跳过")
                else:
                    print("⚠️ 详情页超时")

                tab.close()
                # 因为不调 API，这里 sleep 可以稍微短一点，但为了反爬还是建议 2-3秒
                time.sleep(random.uniform(2.0, 3.5))

            except Exception as e:
                print(f"❌ 错误: {e}")
                try:
                    tab.close()
                except:
                    pass

        print(">>> 本页结束，请继续...")


# ================= 模块二：数据工厂 (摘要 + 向量化) =================
def run_processor():
    print("\n=== 启动数据处理模式 (Qwen摘要 + Ollama向量) ===")
    print("正在扫描未处理的数据 (Embedding 为空的记录)...")

    db = DatabaseManager()

    while True:
        # 1. 获取未处理的任务
        # 只要 embedding 是空的，就说明需要处理（不管 summary 有没有）
        jobs = db.fetch_jobs_without_embedding(limit=10)  # 一次处理 10 条，避免处理太久
        if not jobs:
            print("🎉 所有数据处理完毕！")
            break

        print(f"\n⚡ 本批次处理 {len(jobs)} 条数据...")

        for job in jobs:
            # 解包 tuple
            job_id, title, company, salary, welfare, detail, city, district, experience, degree = job

            print(f"正在处理: {title[:10]}...", end="")
            start_t = time.time()

            # --- 步骤 A: 生成摘要 (Qwen) ---
            # 如果数据库里原本没有摘要，我们现在生成
            # 如果你有旧数据，这里会自动补全摘要
            summary = generate_summary(detail)

            # --- 步骤 B: 生成向量 (Ollama) ---
            # 向量化的内容：标题 + 公司 + 薪资 + 智能摘要 (使用摘要而不是全文，向量质量更高)
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

            # 调用 database.py 里的模型
            # 注意：DatabaseManager 初始化时已经加载了 embed_model
            vector = db.embed_model.embed_query(text_to_embed)

            if not vector:
                print(" ❌ 向量生成失败")
                continue

            # --- 步骤 C: 同时更新回数据库 ---
            # 需要在 database.py 增加 update_job_analysis 方法
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
    print("1. 极速爬虫 (只负责搬运数据，不消耗 Token)")
    print("2. 数据工厂 (后台生成摘要 + 向量化)")
    print("================================================")

    choice = input("请输入功能序号 (1/2): ")

    if choice == '1':
        run_crawler()
    elif choice == '2':
        run_processor()  # 注意这里改名了，叫 processor 更贴切
    else:
        print("无效输入")