import json
import time
import logging
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


class TextDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(json_file, "r", encoding="utf8") as file:
            raw_data = json.load(file)
            for text in raw_data:
                # Tokenize the text and check its length
                tokens = tokenizer.encode(text, add_special_tokens=True)
                if len(tokens) > max_length:
                    # Split the text into chunks
                    for i in range(0, len(tokens), max_length):
                        chunk = tokens[i : i + max_length]
                        self.data.append(tokenizer.decode(chunk))
                else:
                    self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )


def fine_tune(
    pretrained_model_file_path="zephyr-7b-beta",
    training_data="output/arxiv_tex_clean.json",
    lr=5e-5,
    gradient_clip=1.0,
    num_epochs=3,
    out_dir=None,
):
    if out_dir is None:
        raise ValueError("out_dir must be specified.")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logging.basicConfig(
        filename=os.path.join(out_dir, "fine_tune.log"), encoding="utf-8", level=logging.INFO
    )

    # save arguments to log
    logging.info(f"pretrained_model_file_path: {pretrained_model_file_path}")
    logging.info(f"training_data: {training_data}")
    logging.info(f"lr: {lr}")
    logging.info(f"gradient_clip: {gradient_clip}")
    logging.info(f"out_dir: {out_dir}")
    logging.info(f"num_epochs: {num_epochs}")

    logging.info("Starting fine-tuning.")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_file_path)
    dataset = TextDataset(training_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        AutoModelForCausalLM.from_pretrained(pretrained_model_file_path)
        .to(device)
        .to(dtype=torch.bfloat16)
    )

    # optimizer = AdamW(model.parameters(), lr=5e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    start_time = time.time()
    last_save_time = start_time
    save_id = 0
    loss_history = []
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].squeeze(1).to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss_history.append(loss.item())

            # GPU memory usage, batch ID, epoch ID, and loss
            elapsed_time = time.time() - start_time
            total_batches_processed = epoch * len(dataloader) + batch_idx + 1
            estimated_total_time = (
                elapsed_time * num_epochs * len(dataloader) / total_batches_processed
            )
            estimated_time_remaining = estimated_total_time - elapsed_time
            remaining_hours, remaining_rem = divmod(estimated_time_remaining, 3600)
            remaining_minutes, remaining_seconds = divmod(remaining_rem, 60)
            info_string = (
                f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f}, GPU Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
                f"Time Remaining: {int(remaining_hours)}h {int(remaining_minutes)}m {int(remaining_seconds)}s"
            )
            logging.info(info_string)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip
                )
            optimizer.step()

            total_loss += loss.item()

            # for every 10 hours that pass, save one temporary checkpoint
            if time.time() - last_save_time > 36000:
                save_id += 1
                model.save_pretrained(f"{out_dir}/cp_{save_id}")
                last_save_time = time.time()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

    model.save_pretrained(out_dir)

    # save loss history as pickle file
    with open(os.path.join(out_dir, "loss_history.pkl"), "wb") as file:
        pickle.dump(loss_history, file)

    # save tokenizer
    tokenizer.save_pretrained(out_dir)
