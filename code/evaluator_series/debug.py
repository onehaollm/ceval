import argparse

from datasets import load_dataset

from evaluators.deepseek import DeepSeek_Evaluator

dataset=load_dataset(r"ceval/ceval-exam",name="computer_network")

choices = ["A", "B", "C", "D"]

def format_example(self, line, include_answer=True, cot=False):
    example = line['question']
    for choice in self.choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    example += '\n答案：'
    if include_answer:
        if cot:
            ans = line["answer"]
            content = "让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{ans}。"
            return [
                {"role": "user", "content": example},
                {"role": "assistant", "content": content}
            ]
        else:
            return [
                {"role": "user", "content": example},
                {"role": "assistant", "content": line["answer"]}
            ]
    else:
        return [
            {"role": "user", "content": example},
        ]

# print(dataset['val'][0])
# # {'id': 0, 'question': '使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____', 'A': '1', 'B': '2', 'C': '3', 'D': '4', 'answer': 'C', 'explanation': ''}
# format_example(dataset['val'][0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--openai_key", type=str,default="xxx")
    parser.add_argument("--minimax_group_id", type=str,default="xxx")
    parser.add_argument("--minimax_key", type=str,default="xxx")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name",type=str,default="deepseek-r1")
    parser.add_argument("--cot",action="store_true")
    parser.add_argument("--subject","-s",type=str,default="operating_system")
    parser.add_argument("--cuda_device", type=str)
    args = parser.parse_args()
    if "deepseek-r1" in args.model_name:
        evaluator=DeepSeek_Evaluator(
            choices=choices,
            k=args.ntrain,
            api_key=args.openai_key,
            model_name=args.model_name
        )
    print(dataset['val'][0])
    question = evaluator.format_example(dataset['val'][0])
    print(question)
