import os
import random
import json
import requests
from openai import OpenAI
import pdb
# 通过 API 调用远程商业大语言模型（GPT-4、Claude、Gemini）作为受害模型，处理攻击查询并返回响应。
class LLMRequester (object):
    def __init__(self, llm_model):
        self.gpt_api_keys=os.environ.get("OPENAI_API_KEYS")
        self.gpt_api_key_list = self.gpt_api_keys.split(',')
        self.gpt_base_url=os.environ.get("OPENAI_BASE_URL")

        self.claude_api_keys=os.environ.get("ANTHROPIC_API_KEYS")
        self.claude_api_key_list = self.claude_api_keys.split(',')
        self.claude_base_url = os.environ.get("ANTHROPIC_BASE_URL")

        self.gemini_api_keys=os.environ.get("GEMINI_API_KEYS")
        self.gemini_api_key_list = self.gemini_api_keys.split(',')
        self.gemini_base_url = os.environ.get("GEMINI_BASE_URL")

        self.model=llm_model

    def request(self,messages):
        if "gpt" in self.model:
            return self.request_gpt(messages)
        elif "claude" in self.model:
            return self.request_claude(messages)
        else:
            return self.request_gemini(messages)

    def request_gpt(self, messages):
        api_key = random.sample(self.gpt_api_key_list, 1)[0]
        client = OpenAI(api_key=api_key, base_url=self.gpt_base_url)
        completion = client.chat.completions.create(model=self.model, messages=messages)
        response = completion.choices[0].message.content
        return response

    def request_claude(self, messages):
        api_key = random.sample(self.claude_api_key_list, 1)[0]
        url = self.claude_base_url + "/v1/chat/completions"
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {api_key}',
                   'User-Agent': 'Apifox/1.0.0 (https://apifox.com)', 'Content-Type': 'application/json'}
        payload = json.dumps({"model": self.model, "messages": messages})
        response_tmp = requests.request("POST", url, headers=headers, data=payload)
        response = response_tmp.json()
        return response['choices'][0]['message']['content']

    def request_gemini(self, messages):
        api_key = random.sample(self.gemini_api_key_list, 1)[0]
        payload = json.dumps({"model": self.model, "messages": messages})
        url = self.gemini_base_url + "/v1/chat/completions"
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {api_key}',
                   'User-Agent': 'Apifox/1.0.0 (https://apifox.com)', 'Content-Type': 'application/json'}
        response_tmp = requests.request("POST", url, headers=headers, data=payload)
        response = response_tmp.json()
        return response['choices'][0]['message']['content']

