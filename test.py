# from config import API_KEY, BASE_URL, MODEL
# from openai import OpenAI
# from stage_1_function import call_llm_with_retry
# import boolean
import httpie.downloads
import httpie.sessions

if __name__ == "__main__":
    # 1. 先创建 Session 实例
    s = httpie.sessions.Session()
    
    # 2. 再调用实例方法，self 由 Python 自动传，你只传 request_headers
    request_headers = {"User-Agent": "HTTPie"}
    result = s.update_headers(request_headers)
    print(result)