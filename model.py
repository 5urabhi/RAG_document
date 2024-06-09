from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class Model:
    def __init__(self):
        self.model_name = 'microsoft/Phi-3-medium-128k-instruct'
        
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        return tokenizer
