from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM, AutoTokenizer, pipeline
import torch
import copy

from grow import grow_depth, grow_width

def load_model(path, model_name):
    print(f"Loading model...")

    base_model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        # device_map=0,
    )
    model = PeftModel.from_pretrained(base_model, path)
    model = model.merge_and_unload()
    return model


def interpolate(model_1: AutoModelForCausalLM, model_2: AutoModelForCausalLM, alpha: float) -> AutoModelForCausalLM:
    """
    Interpolate between two models with the ratio of weights alpha * model_1 + (1 - alpha) * model_2

    Args:
        model_1: First model
        model_2: Second model
        alpha: Interpolation ratio
    """

    # Get the state_dict of the models
    state_dict_1 = model_1.state_dict()
    state_dict_2 = model_2.state_dict()

    # Interpolate between the weights
    new_state_dict = {}
    for key in state_dict_1:
        new_state_dict[key] = alpha * state_dict_1[key] + (1 - alpha) * state_dict_2[key]

    # Create a copy of the first model and load the new state_dict
    model = copy.deepcopy(model_1)
    model.load_state_dict(new_state_dict)

    return model


def interpolate_models(model_1: PeftModel, model_2: PeftModel, n_interpolations: int, callback: callable):
    """
    Interpolate between two models n_interpolation times and call the callback function with the 
    interpolated model. Define the ratios as equally spaced between 0 and 1 based on n_interpolations.
    For instance, if n_interpolations=5, the ratios will be [0, 0.25, 0.5, 0.75, 1]

    Args:
        model_1: First model
        model_2: Second model
        n_interpolations: Number of interpolations
        callback: Function to call with the interpolated model
    """

    for i in range(n_interpolations):
        alpha = i / (n_interpolations - 1)
        print('-'*50)
        print(f"Interpolating with alpha={alpha}")
        model_interpolated = interpolate(model_1, model_2, alpha)
        callback(model_interpolated)

def generate_example_callback(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str):
    
    if not torch.cuda.is_available():
        model = model.to(torch.float32)

    # Output of model
    print('-'*50)
    print('Model output:')
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = generator(prompt, max_length=128, num_return_sequences=1, truncation=True, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
    print(output)
    

if __name__ == "__main__":

    # Get fine-tuned 70m model
    model_70m = "EleutherAI/pythia-70m"
    path_lora_70m = "models/70m_no_trainer"
    model_70m_ft = load_model(path_lora_70m, model_70m)
    

    # Get fine-tuned 410m model
    model_410m = "EleutherAI/pythia-410m"
    path_lora_410m = "models/410m_no_trainer"
    model_410m_ft = load_model(path_lora_410m, model_410m)

    # Grow 70m model to 410m

    # Depth from 6 -> 24
    intermediate_grown = grow_depth.expand_layers(model_70m_ft, 6, 12, expand_type='alternate')
    intermediate_grown = grow_depth.expand_layers(intermediate_grown, 12, 24, expand_type='alternate')

    # Width from 512 -> 1024
    model_70m_ft_grown_410m = grow_width.expand_width(intermediate_grown, 512, 1024)

    # Define a prompt
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite three advantages of fruits\n\n### Response:\n"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_70m)

    # Define a partial function for the generate_example_callback
    partial_callback = lambda model: generate_example_callback(model, tokenizer, prompt)

    # Interpolate 2 times between 70m and 410m models
    interpolate_models(model_70m_ft_grown_410m, model_410m_ft, 5, partial_callback)
    