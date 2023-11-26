import json
import time
import logging
import pickle
import os
import multiprocessing
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


def tokenize_chunk(chunk_data):
    # helper function for tokenizing a chunk of text
    # has to be outside of the class for multiprocessing (must be picklable)
    chunk, tokenizer, max_length = chunk_data
    tokens = tokenizer.encode(chunk, add_special_tokens=False)
    return [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]


class TextDataset(Dataset):
    """
    Dataset class for tokenizing a large text file.
    """

    def __init__(
        self,
        json_file,
        tokenizer,
        max_length=512,
        cache_path=None,
        num_cpus=None,
        logger=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        if cache_path is None:
            raise ValueError("cache_path must be specified.")

        if os.path.exists(cache_path):
            logger.info("cached tokenization found, loading dataset")
            with open(cache_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            logger.info("no cached tokenization found, tokenizing dataset now")
            with open(json_file, "r", encoding="utf8") as file:
                raw_data = json.load(file)
            chunked_data = []
            for text in raw_data:
                chunk_length = 50000
                chunked_data.extend(
                    [
                        text[i : i + chunk_length]
                        for i in range(0, len(text), chunk_length)
                    ]
                )
            del raw_data  # release the memory

            if num_cpus is None:
                num_cpus = multiprocessing.cpu_count()

            # Prepare data for multiprocessing
            chunk_data = [(chunk, tokenizer, max_length) for chunk in chunked_data]
            del chunked_data  # release the memory

            # Tokenize the data in parallel
            with multiprocessing.Pool(num_cpus) as pool:
                tokenized_chunks = pool.map(tokenize_chunk, chunk_data)
            del chunk_data  # release the memory

            # Flatten and extend self.data
            for chunks in tokenized_chunks:
                self.data.extend(chunks)
            del tokenized_chunks # release the memory

            # Shuffle the chunks
            shuffle(self.data)

            logger.info("Saving tokenized data to cache")
            with open(cache_path, "wb") as f:
                pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = self.data[idx]
        encoded_dict = self.tokenizer.prepare_for_model(
            token_ids,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoded_dict


def fine_tune(
    pretrained_model_file_path="zephyr-7b-beta",
    training_data="output/arxiv_tex_clean.json",
    lr=5e-5,
    gradient_clip=1.0,
    num_epochs=3,
    out_dir=None,
):
    """
    Fine-tune a pretrained model on a dataset using pytorch.
    """

    logging.basicConfig(
        filename=os.path.join(out_dir, "fine_tune.log"),
        filemode="w",
        encoding="utf-8",
        level=logging.INFO,
    )

    logging.info("start of log")

    if out_dir is None:
        raise ValueError("out_dir must be specified.")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save arguments to log
    logging.info(f"pretrained_model_file_path: {pretrained_model_file_path}")
    logging.info(f"training_data: {training_data}")
    logging.info(f"lr: {lr}")
    logging.info(f"gradient_clip: {gradient_clip}")
    logging.info(f"out_dir: {out_dir}")
    logging.info(f"num_epochs: {num_epochs}")

    logging.info("initialize tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_file_path)
    logging.info("tokenize dataset")
    cache_path = os.path.join(out_dir, "tokenized_dataset.pkl")
    dataset = TextDataset(
        training_data, tokenizer, cache_path=cache_path, logger=logging.getLogger()
    )

    logging.info("initialize dataloader")
    dataloader = DataLoader(dataset, batch_size=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")
    logging.info("initialize model")
    model = (
        AutoModelForCausalLM.from_pretrained(pretrained_model_file_path)
        .to(device)
        .to(dtype=torch.bfloat16)
    )
    logging.info("initialize optimizer")
    # optimizer = AdamW(model.parameters(), lr=5e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    start_time = time.time()
    last_save_time = start_time
    save_id = 0
    loss_history = []
    model.train()
    logging.info("start training loop")
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
