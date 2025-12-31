# Example Commands
CUDA_VISIBLE_DEVICES=1,2 python main_sft_dpo.py --attacker_model mistral --adaptor decop_1 --victim_model qwen > sft_dpo_mistral_1_qwen.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,2 python main_sft_dpo.py --attacker_model qwen --adaptor decop_1 --victim_model qwen > sft_dpo_qwen_1_qwen.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1  python main_sft_dpo.py --attacker_model llama3 --adaptor decop_1 --victim_model mistral > sft_dpo_llama3_1_mistral.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,2 python main_sft_dpo.py --attacker_model qwen --adaptor decop_2 --victim_model qwen > sft_dpo_qwen_2_qwen.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,2 python main_sft_dpo.py --attacker_model mistral --adaptor combined --victim_model llama3 > sft_dpo_mistral_com_llama3.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,2 python main_sft_dpo.py --attacker_model qwen --adaptor combined --victim_model llama3 > sft_dpo_qwen_com_llama3.log 2>&1 &



#attacker_model:  ['llama3', 'mistral', 'qwen']
#adaptor: ['decop_1', 'decop_2', 'combined']
#victim_model: ['llama3', 'mistral', 'qwen']