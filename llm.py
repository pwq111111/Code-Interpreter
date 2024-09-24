import logging
import pickle
from openai import OpenAI
import json
from setting import Prompt, API_KEY
from common import install_and_execute, round_values
from faiss_client import FaissClient
from rich import print
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CACHE_FILE = "data/cache.pkl"


class LLM_Client:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)
        self.used_model = "gpt-4o-mini"
        # self.used_model = "gpt-4o-mini"
        self.search_model = FaissClient()
        self.logger = logging.getLogger(__name__)
        self.cache = self.load_cache()

    def save_cache(self):
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(self.cache, f)
        self.logger.info("Cache saved to file")

    def load_cache(self):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            self.logger.info("Cache loaded from file")
            return cache
        except FileNotFoundError:
            self.logger.info("Cache file not found, starting with an empty cache")
            return {}
        except pickle.PickleError as e:
            self.logger.error(f"Error loading cache: {e}")
            return {}

    def user_input_rewrite(self, input_str: str, used_model: str):
        cache_key = f"user_input_rewrite:{input_str}"
        if cache_key in self.cache:
            self.logger.info("Cache hit for user_input_rewrite")
            return self.cache[cache_key]

        messages = [
            {
                "role": "system",
                "content": Prompt.get("user_input_rewrite"),
            },
            {
                "role": "user",
                "content": input_str,
            },
        ]

        self.logger.info("Cache miss for user_input_rewrite, calling OpenAI API")
        completion = self.client.chat.completions.create(model=used_model, messages=messages, temperature=0.1)
        result = completion.choices[0].message.content

        # Store the result in cache and save to file
        self.cache[cache_key] = result
        self.save_cache()
        return result

    def get_knowledge(self, content: str, top_k: int = 3, used_model: str = "gpt-4o-mini"):
        cache_key = f"get_knowledge:{content}:{top_k}:{used_model}:rerank"
        if cache_key in self.cache:
            self.logger.info("Cache hit for get_knowledge")
            return self.cache[cache_key]

        content = self.user_input_rewrite(input_str=content, used_model=used_model)
        self.logger.info("Retrieving knowledge for content")
        score, knowledge = self.search_model.search_rerank(query=content, topk=top_k)

        # Store the result in cache and save to file
        self.cache[cache_key] = knowledge
        self.save_cache()
        return knowledge

    def get_answer(
        self,
        content: str,
        is_knowledge: bool = True,
        llm_model: str = "gpt-4o-mini",
        is_code: bool = True,
    ):
        if not is_code:
            messages = [
                {
                    "role": "system",
                    "content": Prompt.get("system_prompt_answer"),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]
            completion = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages)

            code_string = completion.choices[0].message.content
            return code_string, code_string

        if is_knowledge:
            knowledge_str = self.get_knowledge(content=content)
            system_prompt = Prompt.get("system_prompt_")
            system_prompt = system_prompt.format(knowledge=knowledge_str)
        else:
            system_prompt = Prompt.get("system_prompt")

        max_retries = 3
        attempt = 0
        result = None
        error_message = None
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": content,
            },
        ]
        while attempt < max_retries:

            attempt += 1
            self.logger.info(f"Attempt {attempt} for get_answer")

            if attempt > 1:
                messages.append(
                    {
                        "role": "assistant",
                        "content": reply,
                    }
                )
                user_error_prompt = Prompt.get("error_prompt")
                user_error_prompt = user_error_prompt.format(error_message)
                messages.append(
                    {
                        "role": "user",
                        "content": user_error_prompt,
                    }
                )

            completion = self.client.chat.completions.create(model=llm_model, messages=messages)

            code_string = completion.choices[0].message.content
            match = re.search(r"```python(.*?)```", code_string, re.DOTALL)

            if match:
                code = match.group(1).strip()
                print(code)
            else:
                error_message = "No valid Python code found in the response."
                self.logger.error(error_message)
                reply = code_string
                continue
            result, error_re = install_and_execute(code)

            if result is not None:
                if isinstance(result, str) and ("Error" in result or "error" in result or "no" in result):
                    error_message = result
                    self.logger.error(f"Execution error: {result}")
                    reply = code_string
                    continue
                result = round_values(result)
                return code, str(result)
            else:
                error_message = error_re
                self.logger.error(f"Execution error: {error_message}")
                reply = code_string
                continue

        return code_string, code_string

    def integrate_get_answer(self, content: str):
        code_knowledge, answer_knowledge = self.get_answer(content=content, is_knowledge=True)
        print("Knowledge-based answer:", answer_knowledge)
        # code_no_knowledge, answer_no_knowledge = self.get_answer(
        #     content=content, is_knowledge=False
        # )
        # print("Non-knowledge-based answer:", answer_no_knowledge)
        if "Error" in answer_knowledge or "error" in answer_knowledge or "has no" in answer_knowledge:

            code_nocode, answer_nocode = self.get_answer(
                content=content,
                # is_knowledge=False,
                is_code=False,
            )

            return code_nocode, answer_nocode

        return code_knowledge, answer_knowledge

        judge_answer_system = Prompt.get("judge_answer_system")
        judge_answer_user = Prompt.get("judge_answer_user")

        judge_answer_user = judge_answer_user.format(
            question=content, answer1=answer_knowledge, answer2=answer_no_knowledge
        )
        if answer_knowledge == answer_no_knowledge:
            return code_knowledge, answer_knowledge

        messages = [
            {
                "role": "system",
                "content": judge_answer_system,
            },
            {
                "role": "user",
                "content": judge_answer_user,
            },
        ]

        completion = self.client.chat.completions.create(model=self.used_model, messages=messages, temperature=0.1)

        return_content = completion.choices[0].message.content

        def extract_json_from_function(text):
            # 使用正则表达式匹配 json() 中间的字符串
            json_match = re.search(r"```json(.*?)```", text, re.DOTALL)

            if json_match:
                return json_match.group(1)  # 返回括号中间的字符串
            else:
                print("没有找到json()函数中间的内容。")
                return None

        json_str = extract_json_from_function(return_content)
        print(json_str)
        if json_str:
            try:
                json_data = json.loads(json_str)  # 将字符串解析为字典
            except json.JSONDecodeError:
                print("解析 JSON 数据时出错。")
                return code_knowledge, answer_knowledge

            if isinstance(json_data, dict) and "answer1" in json_data and "answer2" in json_data:
                if json_data["answer1"].get("Reasonableness", 0) == 0:
                    return code_no_knowledge, answer_no_knowledge
                if json_data["answer2"].get("Reasonableness", 0) == 0:
                    return code_knowledge, answer_knowledge
                score1 = sum(json_data.get("answer1", {}).values())
                score2 = sum(json_data.get("answer2", {}).values())

                if score1 > score2:
                    return code_knowledge, answer_knowledge
                else:
                    return code_no_knowledge, answer_no_knowledge
        return code_knowledge, answer_knowledge


if __name__ == "__main__":
    llm_client = LLM_Client()
    content = "As an epidemiologist studying disease patterns and public health issues to prevent and control infectious diseases, could you utilize the 'dcs' (Divide and Conquer Strategy) function in the 'cdlib' library to analyze the transmission patterns of a specific virus based on the data collected from the 'as-22july06.gml' file, and then provide the community detection result in JSON format?"
    answer = llm_client.integrate_get_answer(content=content, use_cached=False)
    print(answer)
