import os
from tqdm import tqdm
import openai
from evaluators.evaluator import Evaluator
from time import sleep
import re


class DeepSeek_Evaluator(Evaluator):
    def __init__(self, choices, k, api_key,model_name,model_api,subset):
        super(DeepSeek_Evaluator, self).__init__(choices, model_name, k)
        openai.api_key = api_key
        self.model_api = model_api
        self.subset = subset

    def format_example(self,line,include_answer=True,cot=False):
        example=line['question']
        for choice in self.choices:
            example+=f'\n{choice}. {line[f"{choice}"]}'

        example+='\n答案：'
        if include_answer:
            if cot:
                ans=line["answer"]
                content="让我们一步一步思考，\n"+line["explanation"]+f"\n所以答案是{ans}。"
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":content}
                ]
            else:
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":line["answer"]}
                ]
        else:
            return [
                {"role":"user","content":example},
            ]
    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt=[
            {
                "role":"system",
                "content":f"你是一个中文人工智能助手，以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。"
            }
        ]
        k=self.k
        if self.k==-1:
            k=dev_df.shape[0]
        for i in range(k):
            tmp=self.format_example(dev_df.iloc[i,:],include_answer=True,cot=cot)
            if i==0:
                tmp[0]["content"]=f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"+tmp[0]["content"]
            prompt+=tmp
        return prompt

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None,cot=False):
        correct_num = 0
        if save_result_dir:
            result = []
            score=[]
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df,cot=cot)
        else:
            few_shot_prompt=[
                {
                    "role":"system",
                    "content":f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。"
                }
            ]
        answers = list(test_df[self.subset]['answer'])
        dataset = test_df[self.subset]
        # for row_index, row in tqdm(dataset.column_names,total=len(dataset)):
        for i in range(len(dataset)):
            row = dataset[i]
            question = self.format_example(row, include_answer=False)

            full_prompt = few_shot_prompt + question
            if not few_shot:
                full_prompt[-1]["content"]=f"以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n\n, 答案输出格式为答案为：A/B/C/D"+full_prompt[-1]["content"]
            response=None
            timeout_counter=0
            while response is None and timeout_counter<=30:
                try:
                    # response = openai.ChatCompletion.create(
                    #     model=self.model_name,
                    #     messages=full_prompt,
                    #     temperature=0.
                    # )
                    response = self.create(few_shot_prompt[0]['content'],question[0]['content'])
                    print("Model output:\n" + response)
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter+=1
                    print(msg)
                    sleep(5)
                    continue
            if response==None:
                response_str=""
            else:
                # response_str = response['choices'][0]['message']['content']
                response_str = response
            #print(response_str)
            if cot:
                ans_list=re.findall(r"答案是(.+?)。",response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"答案为(.+?)。",response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"选项(.+?)是正确的。",response_str)

                if len(ans_list)==0:
                    correct=0
                else:
                    if self.exact_match(ans_list[-1],row["answer"]):
                        correct_num+=1
                        correct=1
                    else:
                        correct=0
            else:
                response_str=response_str.strip()
                if few_shot:
                    if len(response_str)>0:
                        if self.exact_match(response_str,row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
                else:
                    if len(response_str)>0:
                        ans_list=self.extract_ans(response_str)
                        if len(ans_list)>0 and (ans_list[-1]==row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
            if save_result_dir:
                result.append(response_str)
                score.append(correct)
        correct_ratio = 100*correct_num/len(answers)

        if save_result_dir:
            dataset = dataset.add_column(name="model_output", column=result)
            dataset = dataset.add_column(name="correctness", column=score)
            # dataset['model_output']=result
            # dataset["correctness"]=score
            dataset.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'),encoding="utf-8",index=False)
        return correct_ratio

    def extract_ans(self,response_str):
        pattern=[
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
            r"答案.*?[A-D]\.",
            r"答案：*?[A-D]\.",
            r"答案为：*?[A-D]\.",
            r"答案是：*?[A-D]\.",
            r"(?i)正确?答案[：:].*?([A-D])",
            r"(正确|正确选项|答案)[：:]\s*([A-D])",
            r"正确?答案[：:][\s\*]*([A-D])[\.\*]",
            r"(正确|正确选项|答案|答案是|答案为)[：:]\s*([A-D])",
            r"(正确|正确选项|答案|答案是|答案为)[：:]\s*?\*{0,2}([A-D])\.?",
            r"正确?答案[是为][：:][\s*]*([A-D])[\s*.]*",
            r"正确?答案[是为系即][：:][\s*]*([A-D])[\s*.]*",
            r"正确?答案[是为][：:][\s*❖〖]*([A-D])[\s*❖〗.]*",
            r"正确?答案[是为系即][：:]?[\s*]*\*{0,2}([A-D])\*{0,2}[\s*。.]*",
            r"正确?答案[是为系即][：:]?[\s*]*(?:\*\*?)?([A-D])(?:\*\*?)?[\s*。.]*",
        ]
        ans_list=[]
        if response_str[0] in ["A",'B','C','D']:
            ans_list.append(response_str[0])
        for p in pattern:
            if len(ans_list)==0:
                ans_list=re.findall(p,response_str)
            else:
                break
        if len(ans_list) > 0:
            print("**********matched***************")
        else:
            print("**********un-matched***************")
        return ans_list

    def create(self, prompt: str, question: str):
        response = self.model_api(prompt, question)

        return response
