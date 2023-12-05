import time
import logging
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from transformers import AutoConfig
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType


class TokenizedDataset(Dataset):
    """
    Implements a pytorch Dataset for a list of tokenized samples.
    """
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_data[idx])


def fine_tune(
    pretrained_model_file_path="zephyr-7b-beta",
    lr=5e-5,
    gradient_clip=1.0,
    lora_r=1024,  # Rank for LoRA layers
    lora_alpha=32,  # Scaling factor for LoRA
    lora_dropout=0.1,  # Dropout for LoRA
    num_epochs=3,
    out_dir=None,
):
    """
    Fine-tune a pretrained model on a dataset using pytorch.
    """
    if out_dir is None:
        raise ValueError("out_dir must be specified.")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logging.basicConfig(
        filename=os.path.join(out_dir, "fine_tune.log"),
        filemode="w",
        encoding="utf-8",
        level=logging.INFO,
    )
    start_time = time.time()
    logging.info(
        f"start of log. initial time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )
    logging.info(f"pretrained_model_file_path: {pretrained_model_file_path}")
    logging.info(f"lr: {lr}")
    logging.info(f"gradient_clip: {gradient_clip}")
    logging.info(f"out_dir: {out_dir}")
    logging.info(f"num_epochs: {num_epochs}")

    logging.info("load tokenized dataset")
    with open(os.path.join(out_dir, "tokenized_dataset.pkl"), "rb") as file:
        dataset = TokenizedDataset(pickle.load(file))

    logging.info("initialize dataloader")
    batch_size = 6
    torch.manual_seed(42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    del dataset  # free memory

    logging.info("initialize model")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    config = AutoConfig.from_pretrained(
        pretrained_model_file_path, torch_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_config(config)
    model = get_peft_model(model, peft_config)
    model.to(dtype=torch.bfloat16)
    model.print_trainable_parameters()
    logging.info(f"model dtype: {model.dtype}")

    logging.info("initialize optimizer")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, eps=1e-8, betas=(0.9, 0.999)
    )
    loss_history = []
    accelerator = Accelerator()
    dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)
    model.train()  # set model to training mode
    logging.info("start training loop")
    last_save_time = start_time
    save_id = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, input_ids in enumerate(dataloader):
            # Forward pass
            with accelerator.autocast():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            loss_history.append(loss.item())

            # log stats
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
            accelerator.backward(loss)
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip
                )
            optimizer.step()

            total_loss += loss.item()

            # for every 10 hours that pass, save one temporary checkpoint
            if time.time() - last_save_time > 36000:
                save_id += 1
                cp_dir = f"{out_dir}/cp_{save_id}"
                logging.info(f"Checkpoint {save_id}")
                logging.info("Saving checkpoint weights")
                model.save_pretrained(cp_dir)
                logging.info("Saving loss history so far")
                with open(os.path.join(cp_dir, "loss_history.pkl"), "wb") as file:
                    pickle.dump(loss_history, file)
                last_save_time = time.time()

    model.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "loss_history.pkl"), "wb") as file:
        pickle.dump(loss_history, file)


if __name__ == "__main__":
    fine_tune(
        pretrained_model_file_path="Yi-6B",
        lr=5e-4,
        gradient_clip=1.0,
        num_epochs=1,
        out_dir="Yi-6B_textbooks_v3",
    )
