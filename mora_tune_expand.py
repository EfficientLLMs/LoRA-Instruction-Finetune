import torch
import argparse
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import wandb
from accelerate import Accelerator
import utils

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='1.4b')
    parser.add_argument("--seed", type=int, default=1006)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--expanded_adapter", type=str, default='./weight/mora_pythia_expanded_410m_1.4b_txf_r=8/')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default="alpaca_instruction_gpt4")
    args = parser.parse_args()

    args.base_model = f"EleutherAI/pythia-{args.name}"

    if args.scheduler:
        args.wandb_name = f"{args.name}_r={args.rank}_{args.lr}_schedule"
        args.output = f"./weight/pythia_{args.name}_expanded_r={args.rank}_{args.lr}_schedule/"
    else:
        args.wandb_name = f"{args.name}_expanded_r={args.rank}_{args.lr}_fixed"
        args.output = f"./weight/pythia_{args.name}_expanded_r={args.rank}_{args.lr}_fixed/"
        

    # accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"device: {device}")

    # seed
    utils.seed_everything(args.seed)

    # dataloader
    train_dataloader, eval_dataloader = utils.get_dataloader(args.dataset_name, args.batch_size)

    # wandb
    if accelerator.is_main_process:
        wandb.init(
            name=args.wandb_name,
            # project="dump",
            project="mora-instruction-finetune", 
            entity="vibhamasti"
        )
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": train_dataloader.batch_size,
            "seed": args.seed
        }

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,  # standard model; the same tokenizer is used for all models
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    
    # larger base model
    model = GPTNeoXForCausalLM.from_pretrained(
        args.base_model,
        device_map=device,
        use_cache=False,
    )

    # config = LoraConfig(
    #     # enable MoRA
    #     use_mora=True,
    #     # type 1 (Sharing) for large lora ranks, Eq. 6 in paper
    #     # type 6 (RoPE based) for small lora ranks, Eq. 9 in paper
    #     mora_type=6,
    #     # lora rank here, we will calculate corresponding $\hat{r}$ in MoRA
    #     r=args.rank,
    #     # MoRA does not use lora_alpha
    #     # lora_alpha=lora_alpha,
    #     target_modules=["query_key_value"],
    #     lora_dropout=0.05,
    #     task_type="CAUSAL_LM",
    #     # **kwargs,
    # )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.load_adapter(args.expanded_adapter)

    # unfreeze lora weights
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

        # prepare for accelerator
        model, optimizer, scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, eval_dataloader
        )

    else:

        # prepare for accelerator
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
        
        scheduler = None

    # training and save
    utils.train_model(model, optimizer, scheduler, args.lr, train_dataloader, eval_dataloader, args.epochs, accelerator, tokenizer, args.output)
    wandb.finish()