import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# 加载并管理本地部署的受害模型（Victim Models），处理攻击查询并返回模型响应
class LocalLLMRequester(object):
    def __init__(self, base_model):
        #base_model: ["llama3", "mistral", "qwen"]
        self.base_model = base_model
        self.base_model_dict = {"llama3": "path-to-/Meta-Llama-3-8B-Instruct-hf",
                                "mistral": "path-to-/Mistral-7B-Instruct-v0.3",
                                "qwen": "path-to-/qwen2.5-7b-instruct"}
        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count == 1:
            self.device = torch.device("cuda:0")  # Use the only visible GPU
        else:
            self.device = torch.device("cuda:1")  # Default to the second GPU in visible list

        # Load the model and tokenizer
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_dict[self.base_model],
            return_dict=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_dict[self.base_model])
        model = model.to(self.device)
        model.eval()
        return model, tokenizer

    def request(self, messages):
        if self.base_model == "qwen":
            return self.request_qwen(messages)
        elif self.base_model == "mistral":
            return self.request_mistral(messages)
        else:
            return self.request_llama3(messages)

    def request_llama3(self, messages):
        #messages = [{"role": "user", "content": content}]
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = self.model.generate(input_ids, eos_token_id=terminators,max_new_tokens=512,do_sample=True)
        response_tmp = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_tmp, skip_special_tokens=True)
        return response

    def request_mistral(self, messages):
        #messages = [{"role": "user", "content": cont}, ]
        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=512, do_sample=True)
        response_tmp = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response=response_tmp.replace(messages[0]["content"], '').strip()
        return response

    def request_qwen(self, messages):
        #messages = [{"role": "user", "content": cont}, ]
        text = self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(model_inputs.input_ids,max_new_tokens=512,do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

# if __name__=="__main__":
#     local_req=LocalLLMRequester("mistral") #qwen #llama3
#     messages=[{'role': 'user', 'content': 'Hello, Please tell me how can I have more hair?'}]
#     print(local_req.request(messages))