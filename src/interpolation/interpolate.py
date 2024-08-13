from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM, AutoTokenizer, pipeline
import torch
import copy
from datasets import load_metric

import argparse
from typing import Union, List

import eval_utils
from grow import grow_depth, grow_width




class Interpolate:
    """
    Static class with functions to interpolate between two models
    """

    @staticmethod
    def _interpolate_weights_model(
        model_1: AutoModelForCausalLM, 
        model_2: AutoModelForCausalLM, 
        alpha: float
    ) -> AutoModelForCausalLM:
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

        # Interpolate between the weights - only change nn.Linear weights
        new_state_dict = {}
        for key in state_dict_1:
            new_state_dict[key] = alpha * state_dict_1[key] + (1 - alpha) * state_dict_2[key]

        # Create a copy of the first model and load the new state_dict
        model = copy.deepcopy(model_1)
        model.load_state_dict(new_state_dict)

        return model

    @staticmethod
    def _interpolate_logits(
            model_1: AutoModelForCausalLM, 
            model_2: AutoModelForCausalLM, 
            tokenizer: AutoTokenizer, 
            alpha: float, 
            prompt: Union[str, List[str]],
            max_length: int = 128
        ) -> List[str]:
        """
        Run a generation loop till max length and do a weighted sum of the logits of each token
        before doing an argmax to get the token

        Args:
            model_1: First model
            model_2: Second model
            alpha: Interpolation ratio
        """
        # Tokenize the list of prompts
        tokenizer.pad_token = tokenizer.eos_token
        tokenized = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = tokenized['input_ids']
        attn_mask = tokenized['attention_mask']

        gen_tokens_batch = []


        # Distribute inference to multiple GPUs if available
        model_1 = torch.nn.DataParallel(model_1)
        model_2 = torch.nn.DataParallel(model_2)


        for _ in range(max_length):
            model_1_pred = model_1(input_ids=input_ids, attention_mask=attn_mask)
            model_2_pred = model_2(input_ids=input_ids, attention_mask=attn_mask)

            next_token_logits = alpha * model_1_pred[0][:, -1, :] + (1 - alpha) * model_2_pred[0][:, -1, :]

            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append the generated tokens for all batches (batch_size, max_seq_len)
            gen_tokens_batch.append(next_tokens)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            attn_mask = torch.cat([attn_mask, torch.ones_like(attn_mask[:, :1])], dim=-1)

        # Decode the generated tokens
        gen_tokens_batch = torch.stack(gen_tokens_batch, dim=1)[:, :, 0]

        # Pad all tokens after eos with eos
        eos_mask = gen_tokens_batch == tokenizer.eos_token_id
        eos_mask = eos_mask.cumsum(dim=1) > 0

        gen_tokens_batch[eos_mask] = tokenizer.eos_token_id

        generated_text = tokenizer.batch_decode(gen_tokens_batch, skip_special_tokens=True)

        return generated_text

    @staticmethod
    def interpolate_models_logits_rouge(
        model_1: PeftModel, 
        model_2: PeftModel, 
        n_interpolations: int, 
        tokenizer: AutoTokenizer, 
        prompt: str
    ):
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
            generated_texts = Interpolate._interpolate_logits(model_1, model_2, tokenizer, alpha, prompt)
            print(generated_texts)
            # Calculate rouge


    @staticmethod
    def _loss_tracking_callback(model, dataloader, device, tokenizer, losses, rouges):
        model.to(device)
        loss, rouge = eval_utils.evaluate_model_loss_rouge(model, dataloader, device, tokenizer)

        losses.append(loss)
        rouges.append(rouge)

    @staticmethod
    def interpolate_models_by_weights_loss_rouge(
        model_1: PeftModel, 
        model_2: PeftModel, 
        n_interpolations: int,
        dataloader: torch.utils.data.DataLoader, 
        device: torch.device, 
        tokenizer: AutoTokenizer,
    ) -> List[List[float]]:
        losses = []
        alphas = []
        rouges = []
        for i in range(n_interpolations):
            alpha = i / (n_interpolations - 1)
            alphas.append(alpha)
            print('-'*50)
            print(f"Interpolating with alpha={alpha}")
            model_interpolated = Interpolate._interpolate_weights_model(model_1, model_2, alpha)
            Interpolate._loss_tracking_callback(model_interpolated, dataloader, device, tokenizer, losses, rouges)
        return losses, alphas, rouges
    


    @staticmethod
    def interpolate_models_by_logits_loss(
        model_1: PeftModel, 
        model_2: PeftModel, 
        n_interpolations: int,
        dataloader: torch.utils.data.DataLoader, 
        device: torch.device, 
        tokenizer: AutoTokenizer,
        callback: callable
    ):
        losses = []
        alphas = []
        for i in range(n_interpolations):
            alpha = i / (n_interpolations - 1)
            alphas.append(alpha)
            print('-'*50)
            print(f"Interpolating with alpha={alpha}")

        return losses, alphas

if __name__ == "__main__":

    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_small", type=str, default="./output/70m/no_trainer/")
    parser.add_argument("--output_large", type=str, default="./output/410m/no_trainer/")
    parser.add_argument('--base_model_small', type=str, default="EleutherAI/pythia-70m")
    parser.add_argument('--base_model_large', type=str, default="EleutherAI/pythia-410m")
    parser.add_argument('--n_interpolations', type=int, default=5)

    args = parser.parse_args()

    # Get fine-tuned 70m model
    model_70m = "EleutherAI/pythia-70m"
    # path_lora_70m = "models/70m_no_trainer"
    path_lora_70m = "output/70m/no_trainer"
    model_70m_ft = eval_utils.load_model(model_70m, path_lora_70m)
    

    # Get fine-tuned 410m model
    model_410m = "EleutherAI/pythia-410m"
    # path_lora_410m = "models/410m_no_trainer"
    path_lora_410m = "output/410m/no_trainer"
    model_410m_ft = eval_utils.load_model(model_410m, path_lora_410m)

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_70m)

    # Interpolate logits
    # prompt = [
    #     "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite three advantages of fruits\n\n### Response:\n",
    #     "Complete the following sentence: The best way to learn is to \n",
    # ]

    # interpolate_logits(model_70m_ft, model_410m_ft, tokenizer, 0, prompt)
    
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite three advantages of fruits\n\n### Response:\n"
    # interpolate_logits(model_70m_ft, model_410m_ft, tokenizer, 0.5, prompt)

    # interpolate_models_logits(model_70m_ft, model_410m_ft, args.n_interpolations, print, tokenizer, prompt)

    # TODO: remove hard-coding here

    # Depth from 6 -> 24
    # intermediate_grown = grow_depth.expand_layers(model_70m_ft, 6, 12, expand_type='alternate')
    # intermediate_grown = grow_depth.expand_layers(intermediate_grown, 12, 24, expand_type='alternate')

    # # Width from 512 -> 1024
    # model_70m_ft_grown_410m = grow_width.expand_width(intermediate_grown, 512, 1024)

    # print("Model growth complete")

    # # Define a prompt
    # # prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite three advantages of fruits\n\n### Response:\n"

    # # Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_70m)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    # # Define a partial function for the generate_example_callback
    # # partial_callback = lambda model: generate_example_callback(model, tokenizer, prompt)

    # # Interpolate 2 times between 70m and 410m models
    # # interpolate_models_generation(model_70m_ft_grown_410m, model_410m_ft, 5, partial_callback)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # eval_dataloader = torch.load("data/eval_dataloader.pt")
    # losses, alphas = interpolate_models_loss(model_70m_ft_grown_410m, model_410m_ft, 5, eval_dataloader, device, tokenizer)
    # plot_losses(alphas, losses, save_path='interpolation_loss_fine_tuned.png')
    
