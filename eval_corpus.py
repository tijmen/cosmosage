import time
import logging
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from accelerate import Accelerator

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_data[idx])

def evaluate_model(
    pretrained_model_file_path="zephyr-7b-beta",
    out_dir=None,
):
    """
    Evaluate a pretrained model on a dataset using pytorch, computing loss for each sample.
    """
    if out_dir is None:
        raise ValueError("out_dir must be specified.")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logging.basicConfig(
        filename=os.path.join(out_dir, "evaluate_model.log"),
        filemode="w",
        encoding="utf-8",
        level=logging.INFO,
    )
    start_time = time.time()
    logging.info(f"start of log. initial time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    logging.info(f"pretrained_model_file_path: {pretrained_model_file_path}")
    logging.info(f"out_dir: {out_dir}")

    logging.info("load tokenized dataset")
    with open(os.path.join(out_dir, 'tokenized_dataset.pkl'), "rb") as file:
        dataset = TokenizedDataset(pickle.load(file))

    logging.info("initialize dataloader for individual samples")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    logging.info("initialize model")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_file_path, torch_dtype=torch.bfloat16)
    model.eval() # set model to evaluation mode
    logging.info(f"model dtype: {model.dtype}")

    accelerator = Accelerator()
    dataloader, model = accelerator.prepare(dataloader, model)
    sample_loss_history = []

    logging.info("start evaluation loop for individual samples")
    with torch.no_grad():  # No need to track gradients
        for batch_idx, input_ids in enumerate(dataloader):
            # Forward pass
            with accelerator.autocast():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            sample_loss_history.append(loss.item())

            # Log stats
            info_string = f"Sample {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}"
            logging.info(info_string)

    with open(os.path.join(out_dir, "sample_loss_history.pkl"), "wb") as file:
        pickle.dump(sample_loss_history, file)

if __name__ == "__main__":
    evaluate_model(
        pretrained_model_file_path="Yi-6B",
        out_dir="Yi-6B_evaluation"
    )