#Example Commands
CUDA_VISIBLE_DEVICES=3 python main_sft_dpo.py --attacker_model mistral --adaptor decop_1 --victim_model claude-3-5-haiku-20241022 > sft_dpo_qwen_1_claude.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python main_sft_dpo.py --attacker_model qwen --adaptor decop_2 --victim_model claude-3-5-haiku-20241022 > sft_dpo_qwen_2_claude.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python main_sft_dpo.py --attacker_model llama3 --adaptor decop_1 --victim_model gpt-4o-2024-08-06  > sft_dpo_llama3_1_gpt.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python main_sft_dpo.py --attacker_model mistral --adaptor combined --victim_model gpt-4o-2024-08-06  > sft_dpo_mistral_com_gpt.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main_sft_dpo.py --attacker_model mistral --adaptor decop_2 --victim_model gemini-1.5-pro-latest  > sft_dpo_mistral_2_ge.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python main_sft_dpo.py --attacker_model llama3 --adaptor combined --victim_model gemini-1.5-pro-latest  > sft_dpo_llama3_com_ge.log 2>&1 &


#attacker_model:  ['llama3', 'mistral', 'qwen']
#adaptor: ['decop_1', 'decop_2', 'combined']
#victim_model: ['claude-3-5-haiku-20241022', 'gpt-4o-2024-08-06', 'gemini-1.5-pro-latest']