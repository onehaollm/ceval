import argparse
import os
import time

from datasets import load_dataset

from models.deepseek_openapi import DeepSeek_OpenaiAPI
from evaluators.deepseek import DeepSeek_Evaluator

dataset=load_dataset(r"ceval/ceval-exam",name="computer_network")

choices = ["A", "B", "C", "D"]


# print(dataset['val'][0])
# # {'id': 0, 'question': '使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____', 'A': '1', 'B': '2', 'C': '3', 'D': '4', 'answer': 'C', 'explanation': ''}
# format_example(dataset['val'][0])

if __name__ == "__main__":
    choices = ["A", "B", "C", "D"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--openai_key", type=str, default="sk")
    parser.add_argument("--openai_base_url", type=str, default="https://cloud.infini-ai.com/maas/v1")
    parser.add_argument("--minimax_group_id", type=str, default="xxx")
    parser.add_argument("--minimax_key", type=str, default="xxx")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name", type=str, default="deepseek-r1")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--subject", "-s", type=str, default="operating_system")
    parser.add_argument("--cuda_device", type=str)
    parser.add_argument("--subset", type=str, default="dev")
    args = parser.parse_args()

    subject_name = args.subject
    if not os.path.exists(r"logs"):
        os.mkdir(r"logs")
    run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_result_dir = os.path.join(r"logs", f"{args.model_name}_{run_date}")
    os.mkdir(save_result_dir)
    print(subject_name)
    # val_file_path=os.path.join('data/val',f'{subject_name}_val.csv')
    # val_df=pd.read_csv(val_file_path)

    data_example = {
        "id": 0,
        "question": "使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____ ', 'A': '1', 'B': '2', 'C': '3', 'D': '4'",
        "answer": "C", "explanation": ""
    }

    test_df = load_dataset(r"ceval/ceval-exam", name=subject_name)

    if "deepseek-r1" in args.model_name:
        evaluator = DeepSeek_Evaluator(
            choices=choices,
            k=args.ntrain,
            api_key=args.openai_key,
            model_name=args.model_name,
            model_api=DeepSeek_OpenaiAPI(args.openai_key, args.openai_base_url, model_name=args.model_name,
                                         temperature=0.6, max_tokens=4096),
            subset=args.subset
        )

    dataset = test_df[args.subset]
    score = [1, 0, 1, 0, 1]
    # test_df["correctness"] = score
    # test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'), encoding="utf-8", index=False)


    dataset.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'), encoding="utf-8", index=False)
    correct_ratio = evaluator.eval_subject(subject_name, test_df, few_shot=args.few_shot,
                                           save_result_dir=save_result_dir)
    print("Acc:", correct_ratio)
