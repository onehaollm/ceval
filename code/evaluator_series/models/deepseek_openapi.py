# -*- coding: utf-8 -*-
import requests
import base64

import time
import openai
from random import choice
from typing import List
from openai import OpenAI


class DeepSeek_OpenaiAPI:
    def __init__(self, api_key, base_url, model_name, temperature, max_tokens):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def forward(self, prompt, question):
        usr_content = [{"type": "text","text": question}]

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": prompt}]
                        },
                        {   
                            "role": "user",
                            "content": usr_content
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                break
            except openai.BadRequestError as e:
                print('BadRequestError:', e)
                response = "Your input image may contain content that is not allowed by our safety system."
                break
            except Exception as e:
                print('Exception:', e)
                time.sleep(1)
                continue
        
        return response

    def postprocess(self, response):
        """
        """
        model_output = None
        
        if isinstance(response, str):
            model_output = response
        else:
            model_output = response.choices[0].message.content
        return model_output
    
    def __call__(self, prompt:str, question:str):
        response = self.forward(prompt, question)
        model_output = self.postprocess(response)
        return model_output

def test(model, prompt:str, question:str):
    response = model(prompt, question)

    return response


if __name__ == "__main__":

    openai_key = ""
    openai_base_url = "https://cloud.infini-ai.com/maas/v1"

    model_api = DeepSeek_OpenaiAPI(openai_key, openai_base_url, model_name="deepseek-r1", temperature=0.6, max_tokens=4096)
    data_example = {
            "id": 0, "question": "使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____ ', 'A': '1', 'B': '2', 'C': '3', 'D': '4'", "answer": "C", "explanation": ""
        }
    choice_question = data_example['question']
    subject_name = "operating_system"
    # choice_picture = data_example['picture']
    choice_prompt = f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。"

    result = test(model_api, choice_prompt, choice_question)

    print("Model output:\n" + result)
    
