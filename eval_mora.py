from transformers import GPTNeoXForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
import torch
import torch.nn as nn
from utils import evaluate_model
from accelerate import Accelerator
import pandas as pd
import argparse
import os


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--large_model", type=str, default="1.4b")
    parser.add_argument("--small_model", type=str, default="410m")
    parser.add_argument("--small_adapter", type=str, default="./weight/mora_pythia_410m_r=8_0.0001")
    parser.add_argument("--large_adapter", type=str, default="./weight/mora_pythia_expanded_410m_1.4b_txf_r=8")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--large_rank", type=int, default=4)
    parser.add_argument("--mora_type", type=int, default=6)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_dataloader", type=str, default="./data/eval_dataloader.pt")
    parser.add_argument("--output_csv", type=str, default="./eval/evaluate_mora_410m_1.4b_expand_txf.csv")
    
    args = parser.parse_args()
    
    accelerator = Accelerator()
    
    eval_dataloader = torch.load(args.eval_dataloader)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-" + args.large_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    small_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-" + args.small_model)
    small_model.load_adapter(args.small_adapter)

    mora_config = LoraConfig(
        # enable MoRA
        use_mora=True,
        # type 1 (Sharing) for large lora ranks, Eq. 6 in paper
        # type 6 (RoPE based) for small lora ranks, Eq. 9 in paper
        mora_type=args.mora_type,
        # lora rank here, we will calculate corresponding $\hat{r}$ in MoRA
        r=args.large_rank,
        # MoRA does not use lora_alpha
        # lora_alpha=lora_alpha,
        target_modules=["query_key_value"],
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM"
    )


    large_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-" + args.large_model)
    large_model = get_peft_model(large_model, mora_config)

    eval_results = []

    # Prepare for accelerator
    eval_dataloader, large_model, small_model, tokenizer = accelerator.prepare(
        eval_dataloader, large_model, small_model, tokenizer
    )

    # Evaluate the small fine-tuned model
    eval_loss, eval_rouge_scores = evaluate_model(small_model, eval_dataloader, accelerator, tokenizer)

    eval_results.append(
        {
            "model": "fine_tuned_" + args.small_model,
            "rank": args.rank,
            "eval_loss": eval_loss,
            **eval_rouge_scores,
        }
    )

    # Evaluate the large pre-trained model
    eval_loss, eval_rouge_scores = evaluate_model(large_model, eval_dataloader, accelerator, tokenizer)

    eval_results.append(
        {
            "model": "raw_" + args.large_model,
            "rank": args.rank,
            "eval_loss": eval_loss,
            **eval_rouge_scores,
        }
    )


    # Evaluate the expanded large model

    del large_model

    large_model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-" + args.large_model)
    large_model.load_adapter(args.large_adapter)

    large_model = accelerator.prepare(large_model)

    eval_loss, eval_rouge_scores = evaluate_model(large_model, eval_dataloader, accelerator, tokenizer)

    eval_results.append(
        {
            "model": "expanded_txf_" + args.large_model + "_from_" + args.small_model,
            "rank": args.rank,
            "eval_loss": eval_loss,
            **eval_rouge_scores,
        }
    )

    # Print the evaluation results
    print(eval_results)

    # Save the evaluation results dict to a CSV file
    df = pd.DataFrame(eval_results)

    
    output_folder = os.path.dirname(args.output_csv)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df.to_csv(args.output_csv, index=False)