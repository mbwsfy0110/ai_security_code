from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

def load_and_save_model_adapter(base_model, adaptor): #实现适配器与基础模型的合并。
    # Load the base model in FP16
    model_tmp = AutoModelForCausalLM.from_pretrained(
        base_model_dict[base_model],
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_dict[base_model])
    sft_adaptor_path = sft_adaptor_path_dict[base_model][adaptor]
    # Load the adapter and merge it into the model
    model = PeftModel.from_pretrained(model_tmp, sft_adaptor_path)
    model = model.merge_and_unload()

    # Save the merged model and tokenizer
    save_directory =sft_merge_save_dir[base_model][adaptor]
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Merged model and tokenizer saved to {save_directory}")
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")

if __name__=="__main__":
    base_model_dict = {"llama3": "Your-Downloaded-from-meta-llama/Meta-Llama-3-8B-Instruct",
                       "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
                       "qwen": "Qwen/Qwen2.5-7B-Instruct"}

    sft_adaptor_path_dict = {
        "llama3": {"decop_1": "Yiyiyi/sft_llama3_decop_1",
                   "decop_2": "Yiyiyi/sft_llama3_decop_2",
                   "combined": "Yiyiyi/sft_llama3_combined"},
        "mistral": {"decop_1": "Yiyiyi/sft_mistral_decop_1",
                    "decop_2": "Yiyiyi/sft_mistral_decop_2",
                    "combined": "Yiyiyi/sft_mistral_combined"},
        "qwen": {"decop_1": "Yiyiyi/sft_qwen_decop_1",
                 "decop_2": "Yiyiyi/sft_qwen_decop_2",
                 "combined": "Yiyiyi/sft_qwen_combined"},
    }

    sft_merge_save_dir = {
        "llama3": {"decop_1": "Your-Merged-Model-Path/sft_llama3/decop_1",
                   "decop_2": "Your-Merged-Model-Path/sft_llama3/decop_2",
                   "combined": "Your-Merged-Model-Path/sft_llama3/combined"},
        "mistral": {"decop_1": "Your-Merged-Model-Path/sft_mistral/decop_1",
                    "decop_2": "Your-Merged-Model-Path/sft_mistral/decop_2",
                    "combined": "Your-Merged-Model-Path/sft_mistral/combined"},
        "qwen": {"decop_1": "Your-Merged-Model-Path/sft_qwen/decop_1",
                 "decop_2": "Your-Merged-Model-Path/sft_qwen/decop_2",
                 "combined": "Your-Merged-Model-Path/sft_qwen/combined"},
    }
    #please first make sure the merge_dirs exist

    device = "cuda"
    # if you only want to merge one model
    # base_model='mistral'
    # adaptor='combined'
    # load_and_save_model_adapter(base_model, adaptor)

    # merge all models
    # for base_model in ['mistral','qwen','llama3']:
    #     for adaptor in ['decop_1', 'decop_2','combined']:
    #         load_and_save_model_adapter(base_model, adaptor)