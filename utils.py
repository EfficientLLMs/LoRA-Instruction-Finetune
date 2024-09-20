import random
import os
import torch
from transformers import default_data_collator
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import wandb


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataloader(dataset_name, batch_size=4):

    if dataset_name == "alpaca_instruction_gpt4":

        train_ds_packed = torch.load("./data/train_ds_packed.pt")
        eval_ds_packed = torch.load("./data/eval_ds_packed.pt")

        train_dataloader = DataLoader(
            train_ds_packed,
            batch_size=batch_size,
            collate_fn=default_data_collator,
        )
        eval_dataloader = DataLoader(
            eval_ds_packed,
            batch_size=batch_size,
            collate_fn=default_data_collator,
            shuffle=False,
        )
        

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataloader, eval_dataloader


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


def train_model(model, optimizer, scheduler, learning_rate, train_dl, test_dl, epochs, accelerator, tokenizer, output):
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

        avg_train_loss = total_loss / len(train_dl)
        if accelerator.is_main_process:
            print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss}")

        eval_loss, eval_rouge_scores = evaluate_model(model, test_dl, accelerator, tokenizer)
        if accelerator.is_main_process:

            lr = scheduler.get_last_lr()[0] if scheduler is not None else learning_rate
            wandb.log({
                "epoch": epoch, 
                "learning_rate": lr,
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




