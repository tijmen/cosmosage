import time
import os
import logging
import random
from datasets import load_dataset

class QuantAutoGPTQ:
    def __init__(self, model_name_or_path, output_dir, dataset,
                 num_samples=128, trust_remote_code=False, cache_examples=True,
                 use_fast=True, use_triton=False, bits=[4], group_size=[128], damp=[0.01],
                 desc_act=[False], dtype='float16', seqlen=2048, batch_size=1, stop_file=None,
                 make_folder=False, GPU=0, cuda_alloc_conf=None):

        self.pretrained_model_dir = model_name_or_path
        self.output_dir_base = output_dir
        self.dataset = dataset
        self.num_samples = num_samples
        self.trust_remote_code = trust_remote_code
        self.cache_examples = cache_examples
        self.use_fast = use_fast
        self.use_triton = use_triton

        def check_list(item):
            return item if isinstance(item, list) else [item]

        self.bits = check_list(bits)
        self.group_size = check_list(group_size)
        self.desc_act = check_list(desc_act)
        self.damp = check_list(damp)

        self.dtype = dtype
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.stop_file = stop_file 
        self.make_folder = make_folder

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

        from transformers import AutoTokenizer
        self.logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_dir, 
                                                       use_fast=self.use_fast, 
                                                       trust_remote_code=self.trust_remote_code)

    #TODO: Add support for other datasets and make this more generic
    def get_wikitext2(self):
        import numpy as np
        import torch
        wikidata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        wikilist = [' \n' if s == '' else s for s in wikidata['text'] ]

        text = ''.join(wikilist)
        self.logger.info("Tokenising wikitext2")
        trainenc = self.tokenizer(text, return_tensors='pt')

        random.seed(0)
        np.random.seed(0)
        torch.random.manual_seed(0)

        traindataset = []
        traindataset = []

        for _ in range(self.num_samples):
            i = random.randint(0, trainenc.input_ids.shape[1] - self.seqlen - 1)
            j = i + self.seqlen
            inp = trainenc.input_ids[:, i:j]
            attention_mask = torch.ones_like(inp)
            traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
        return traindataset

    def get_c4(self):
        import numpy as np
        import torch
        traindata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', use_auth_token=False
        )

        trainloader = []
        for _ in range(self.num_samples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = self.tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= self.seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - self.seqlen - 1)
            j = i + self.seqlen
            inp = trainenc.input_ids[:, i:j]
            attention_mask = torch.ones_like(inp)
            trainloader.append({'input_ids':inp,'attention_mask': attention_mask})

        return trainloader

    def quantize(self, output_dir, traindataset, bits, group_size, desc_act, damp):
        # Hide the super annoying bitsandbytes loading message. We don't even use BnB but I don't know if I can stop it loading entirely.
        os.environ['BITSANDBYTES_NOWELCOME'] = '1'

        # We only import Torch and AutoGPTQ when needed, so that earlier set env vars will affect them.
        import torch
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            damp_percent=damp
        )

        if self.dtype == 'float16':
            torch_dtype  = torch.float16
        elif self.dtype == 'float32':
            torch_dtype  = torch.float32
        elif self.dtype == 'bfloat16':
            torch_dtype  = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        self.logger.info(f"Loading model from {self.pretrained_model_dir} with trust_remote_code={self.trust_remote_code} and dtype={torch_dtype}")
        model = AutoGPTQForCausalLM.from_pretrained(self.pretrained_model_dir, quantize_config=quantize_config,
                                                    low_cpu_mem_usage=True, torch_dtype=torch_dtype, trust_remote_code=self.trust_remote_code)

        self.logger.info(f"Starting quantization to {output_dir} with use_triton={self.use_triton}")
        start_time = time.time()
        model.quantize(traindataset, use_triton=self.use_triton, batch_size=self.batch_size, cache_examples_on_gpu=self.cache_examples)

        self.logger.info(f"Time to quantize model at {output_dir} with use_triton={self.use_triton}: {time.time() - start_time:.2f}")

        self.logger.info(f"Saving quantized model to {output_dir}")
        model.save_quantized(output_dir, use_safetensors=True)
        self.logger.info("Done.")

    def run_quantization(self):
        if self.dataset == 'wikitext':
            traindataset = self.get_wikitext2()
        elif self.dataset == 'c4':
            traindataset = self.get_c4()
        else:
            self.logger.error(f"Unsupported dataset: {self.dataset}")
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        abort = False
        iterations=[]
        for bits in self.bits:
            for group_size in self.group_size:
                for desc_act in self.desc_act:
                    for damp in self.damp:
                        desc_act = desc_act == 1 and True or False
                        iterations.append({"bits": bits, "group_size": group_size, "desc_act": desc_act, "damp": damp})

        num_iters = len(iterations)
        if num_iters > 1:
            logger.info(f"Starting {num_iters} quantizations.")
        count=1
        for iteration in iterations:
            if abort:
                break
            if self.stop_file is not None and os.path.exists(self.stop_file):
                self.logger.info(f"Stopping as {self.stop_file} exists")
                abort = True
                break

            bits = iteration['bits']
            group_size = iteration['group_size']
            desc_act = iteration['desc_act']
            damp = iteration['damp']

        try:
            if self.make_folder:
                output_dir = os.path.join(self.output_dir_base, f"{bits}bits-{group_size}g-desc_act_{desc_act}-damp_{damp}")
            else:
                output_dir = self.output_dir_base
            os.makedirs(output_dir, exist_ok=True)
            try:
                if num_iters > 1:
                    self.logger.info(f"Starting quantization {count}/{num_iters}")
                self.logger.info(f"Quantising with bits={bits} group_size={group_size} desc_act={desc_act} damp={damp} to {output_dir}")
                self.quantize(output_dir, traindataset, bits, group_size, desc_act, damp)
            except KeyboardInterrupt:
                logger.error(f"Aborted. Will delete {output_dir}")
                os.rmdir(output_dir)
                abort = True
            except:
                raise

        finally:
            count += 1

if __name__ == "__main__":
    import argparse
    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s %(levelname)s [%(name)s] %(message)s", 
                        level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description='AutoGPTQ quantize')
    parser.add_argument('pretrained_model_dir', type=str, help='Repo name')
    parser.add_argument('output_dir_base', type=str, help='Output base folder')
    parser.add_argument('dataset', type=str, help='Quantisation dataset')
    parser.add_argument('--num_samples', type=int, default=128, help='Number of dataset samples')
    parser.add_argument('--trust_remote_code', action="store_true", help='Trust remote code')
    parser.add_argument('--cache_examples', type=int, default=1, help='Cache examples on GPU')
    parser.add_argument('--use_fast', action="store_true", help='Use fast tokenizer')
    parser.add_argument('--use_triton', action="store_true", help='Use Triton for quantization')
    parser.add_argument('--bits', type=int, nargs='+', default=[4], help='Quantize bit(s)')
    parser.add_argument('--group_size', type=int, nargs='+', default=[128], help='Quantize group size(s)')
    parser.add_argument('--damp', type=float, nargs='+', default=[0.01], help='Quantize damp_percent(s)')
    parser.add_argument('--desc_act', type=int, nargs='+', default=[0], help='Quantize desc_act(s) - 1 = True, 0 = False')
    parser.add_argument('--dtype', type=str, choices=['float16', 'float32', 'bfloat16'], default='float16', help='Unquantised model dtype')
    parser.add_argument('--seqlen', type=int, default=2048, help='Model sequence length')
    parser.add_argument('--batch_size', type=int, default=1, help='Quantize batch size for processing dataset samples')
    parser.add_argument('--stop_file', type=str, help='Filename to look for to stop inference, specific to this instance')
    parser.add_argument('--make_folders', action="store_true", help='Make folders for each quantization using params in folder name')

    args = parser.parse_args()
    quantizer = QuantAutoGPTQ(args.pretrained_model_dir,
                              args.output_dir_base,
                              args.dataset,
                              num_samples=args.num_samples,
                              trust_remote_code=args.trust_remote_code,
                              cache_examples=args.cache_examples,
                              use_fast=args.use_fast,
                              use_triton=args.use_triton,
                              bits=args.bits,
                              group_size=args.group_size,
                              desc_act=args.desc_act,
                              damp=args.damp,
                              dtype=args.dtype,
                              seqlen=args.seqlen,
                              batch_size=args.batch_size,
                              stop_file=args.stop_file,
                              make_folder=args.make_folders)
    quantizer.run_quantization()
