import os
import json
from utils.evaluate_harmfulness import HarmClassifier
from tqdm import tqdm


# 负责加载json格式的交互记录文件
def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# 将评估结果保存为json文件
def save_results(results, output_file):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

# 负责批量计算攻击成功率ASR
def main():
    hc = HarmClassifier()
    for type in ['decop_1','decop_2', 'combined']:
        for attacker in ['llama3', 'mistral', 'qwen']:
            for victim in ['llama3', 'mistral', 'qwen']:
            #for victim in ['gpt-4o-2024-08-06','claude-3-5-haiku-20241022','gemini-1.5-pro-latest']:
                res_dir_path = f"interactions_sft_dpo_attacker_{attacker}_adaptor_{type}_victim_{victim}"
                results_file = f"{res_dir_path}_results.json"  # 每个路径单独对应一个结果文件

                if os.path.exists(results_file):
                    with open(results_file, "r") as f:
                        results = json.load(f)
                else:
                    results = {}

                for i in tqdm(range(520)):
                    response_tmp = os.path.join(res_dir_path, f"{i}.json")
                    if f"{i}" in results:
                        continue
                    try:
                        data = load_json(response_tmp)
                        response = data[0]['Interactions'][-1]['content']
                        flag_llm = hc.classify_with_llm_judge(response)
                        flag_kw = hc.classify_with_kw(response)
                        results[f"{i}"] = {
                            "flag_llm": flag_llm,
                            "flag_kw": flag_kw,
                            "status": "success"
                        }
                    except Exception as e:
                        continue
                    finally:
                        save_results(results, results_file)
                if len(results)==520:
                    print(results_file,len(results))
                    total_count = len(results)

                    flag_llm_true_count = sum(1 for item in results.values() if item["flag_llm"])
                    flag_kw_true_count = sum(1 for item in results.values() if item["flag_kw"])


                    flag_llm_true_ratio = flag_llm_true_count / total_count
                    flag_kw_true_ratio = flag_kw_true_count / total_count

                    print(f"flag_llm ASR: {flag_llm_true_ratio:.3f}")
                    print(f"flag_kw ASR: {flag_kw_true_ratio:.3f}")


if __name__=="__main__":
    main()