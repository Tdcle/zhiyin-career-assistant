from __future__ import annotations

MEMORY_CITY_CANDIDATES = [
    "北京", "上海", "深圳", "广州", "杭州", "成都", "武汉", "西安", "南京", "苏州", "天津", "重庆",
]

MEMORY_ROLE_TOKENS = {
    "前端": ["前端", "frontend", "front-end", "web前端"],
    "后端": ["后端", "backend", "back-end", "服务端"],
    "全栈": ["全栈", "fullstack", "full-stack"],
    "测试": ["测试", "qa", "测试开发"],
    "运维": ["运维", "devops", "sre"],
    "产品": ["产品", "产品经理", "pm"],
    "数据": ["数据分析", "数据开发", "bi", "etl"],
    "算法": ["算法", "机器学习", "深度学习", "cv", "nlp"],
    "AI": ["ai", "大模型", "llm", "agent", "人工智能"],
}

MEMORY_SKILL_TOKENS = [
    "java", "python", "golang", "go", "c++", "c#", "php",
    "vue", "react", "typescript", "javascript", "node", "spring",
    "django", "flask", "fastapi", "mysql", "postgresql", "redis",
    "kafka", "docker", "kubernetes", "pytorch", "tensorflow",
]

MEMORY_CANONICAL_FACT_KEYS = {
    "desired_role",
    "desired_city",
    "desired_experience",
    "desired_salary",
    "core_skill",
    "experience_note",
    "location_note",
    "salary_note",
    "education_note",
    "skill_note",
    "preference_note",
    "legacy_profile",
}

MEMORY_FACT_KEY_ALIASES = {
    "desired_role": "desired_role",
    "desired_roles": "desired_role",
    "role": "desired_role",
    "role_preference": "desired_role",
    "desired_city": "desired_city",
    "desired_cities": "desired_city",
    "city": "desired_city",
    "city_preference": "desired_city",
    "location": "desired_city",
    "desired_experience": "desired_experience",
    "experience_requirement": "desired_experience",
    "desired_salary": "desired_salary",
    "salary": "desired_salary",
    "salary_expectation": "desired_salary",
    "core_skill": "core_skill",
    "core_skills": "core_skill",
    "skill_core": "core_skill",
    "experience_note": "experience_note",
    "experience": "experience_note",
    "location_note": "location_note",
    "location_preference": "location_note",
    "salary_note": "salary_note",
    "salary_preference": "salary_note",
    "education_note": "education_note",
    "education": "education_note",
    "skill_note": "skill_note",
    "skill": "skill_note",
    "preference_note": "preference_note",
    "preference": "preference_note",
    "legacy_profile": "legacy_profile",
}

MEMORY_SINGLE_VALUE_FACT_KEYS = {
    "desired_experience",
    "desired_salary",
}

MEMORY_RESOLVABLE_FACT_KEYS = {
    "desired_role",
    "desired_city",
    "desired_experience",
    "desired_salary",
    "preference_note",
    "location_note",
    "experience_note",
    "salary_note",
}

MEMORY_AUTO_TTL_DAYS = {
    "desired_role": 180,
    "desired_city": 180,
    "desired_experience": 120,
    "desired_salary": 120,
    "preference_note": 60,
    "location_note": 90,
    "experience_note": 90,
    "salary_note": 90,
}

MEMORY_STALE_DEACTIVATE_DAYS = 240
