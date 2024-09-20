import random
import os
import torch
import argparse
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import evaluate
import wandb
from accelerate import Accelerator


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_model(model, test_dl, accelerator, tokenizer):
    model.eval()
    total_loss = 0
    rouge = evaluate.load('rouge')

    if accelerator.is_main_process:
        test_iter = tqdm(test_dl, desc="Evaluating")
    else:
        test_iter = test_dl

    for batch in test_iter:
        labels = batch["labels"].detach().cpu().numpy()

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            generated_ids = torch.argmax(outputs.logits, dim=-1)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            rouge.add_batch(predictions=generated_texts, references=reference_texts)
    
    avg_eval_loss = total_loss / len(test_dl)
    final_rouge_scores = rouge.compute()

    if accelerator.is_main_process:
        print(f"Average Evaluation Loss: {avg_eval_loss}")
        print("ROUGE Scores:", final_rouge_scores)

    return avg_eval_loss, final_rouge_scores

def train_model(model, optimizer, scheduler, lr, train_dl, test_dl, epochs, accelerator, tokenizer, output):
    model.train()
    if accelerator.is_main_process:
        print("Start training...")

    for epoch in tqdm(range(epochs)):
        if accelerator.is_main_process:
            train_iter = tqdm(train_dl, desc=f"Epoch {epoch + 1}")
        else:
            train_iter = train_dl
            
        total_loss = 0
        for batch in train_iter:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()
            learning_rate = scheduler.get_last_lr()[0]
        
        else:
            learning_rate = lr

        avg_train_loss = total_loss / len(train_dl)
        if accelerator.is_main_process:
            print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss}")

        eval_loss, eval_rouge_scores = evaluate_model(model, test_dl, accelerator, tokenizer)
        if accelerator.is_main_process:
            wandb.log({
                "epoch": epoch, 
                "learning_rate": learning_rate,
                "training_loss": avg_train_loss,
                "evaluation_loss": eval_loss, 
                **eval_rouge_scores
            })
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )

    if accelerator.is_main_process:
        print("Training finished...")


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_model", type=str, default='410m')
    parser.add_argument("--large_model", type=str, default='1.4b')
    parser.add_argument("--expand_method", type=str, default='padding')  # copy or padding
    parser.add_argument("--seed", type=int, default=1006)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", action="store_true")
    args = parser.parse_args()

    args.base_model = f"EleutherAI/pythia-{args.large_model}"  # large model

    if args.scheduler:
        args.wandb_name = f"{args.small_model}_{args.large_model}_{args.expand_method}_r=64_{args.lr}_schedule"  
        args.output = f"./weight/pythia_{args.small_model}_{args.large_model}_{args.expand_method}_r=64_{args.lr}_schedule/"
        args.large_adapter = f"./weight/pythia_{args.small_model}_{args.large_model}_{args.expand_method}_r=64_schedule/"
    
    else:
        args.wandb_name = f"{args.small_model}_{args.large_model}_{args.expand_method}_r=8_{args.lr}"  
        args.output = f"./weight/pythia_{args.small_model}_{args.large_model}_{args.expand_method}_r=8_{args.lr}_fixed/"
        args.large_adapter = f"./weight/pythia_expanded_{args.small_model}_{args.large_model}_{args.expand_method}_r=8/"


    # accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"device: {device}")

    # seed
    seed_everything(args.seed)

    # dataloader
    train_dataloader = torch.load("./data/train_dataloader.pt")
    eval_dataloader = torch.load("./data/eval_dataloader.pt")

    # wandb
    if accelerator.is_main_process:
        wandb.init(
            name=args.wandb_name,
            project="lora-instruction-finetune", 
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
    base_model = GPTNeoXForCausalLM.from_pretrained(
        args.base_model,
        device_map=device,
        use_cache=False,
    )
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # add expanded adapter
    model = PeftModel.from_pretrained(base_model, args.large_adapter)

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
        scheduler = None

        # prepare for accelerator
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

    # training and save
    train_model(model, optimizer, scheduler, args.lr, train_dataloader, eval_dataloader, args.epochs, accelerator, tokenizer, args.output)
    wandb.finish()