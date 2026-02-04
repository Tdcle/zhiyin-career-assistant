# utils/global_state.py

class GlobalState:
    """
    用于在 Tool 和 Gradio App 之间传递临时数据
    Key: user_id (或 query)
    Value: 结构化的职位列表 [{'job_id':..., 'title':...}, ...]
    """
    _search_cache = {}

    @classmethod
    def set_search_results(cls, key, results):
        cls._search_cache[key] = results

    @classmethod
    def get_search_results(cls, key):
        return cls._search_cache.get(key, [])

    @classmethod
    def clear(cls, key):
        if key in cls._search_cache:
            del cls._search_cache[key]