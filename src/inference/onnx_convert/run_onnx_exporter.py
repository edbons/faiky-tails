import argparse
import logging
import os
import sys

import numpy as np
import torch

import onnxruntime
import transformers
from gpt3_onnx.generation_onnx import GPT2BeamSearchSampleGenerator
from transformers import GPT2LMHeadModel, GPT2Tokenizer


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Export Sber GPT3 model + Beam Search with Sampling to ONNX graph.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=5,
        help=("The maximum total input sequence length after tokenization."),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        help="Path to pretrained model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--model_tokenizer_path",
        type=str,
        help="Path to pretrained model tokenizer",
        required=True,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device where the model will be run",
    )
    parser.add_argument("--output_file_path", type=str, default=None, help="Where to store the final ONNX file.")
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top p",
    )

    args = parser.parse_args()
    if args.num_beams and args.num_beams < 2:
        parser.error("Minimum num_beams is 2")

    return args


def load_model_tokenizer(chk_path, tokenizer, device="cpu"):
    
    with open(os.path.join(chk_path,'checkpoint.pt'), 'rb') as f:
        huggingface_model = torch.load(f, map_location=device)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)   
    return huggingface_model, tokenizer


def export_and_validate_model(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, onnx_file_path: str, num_beams: int, max_length: int, temperature: float, top_p: float):
    
    # add special tokens to model config    
    padtok =  0
    septok = tokenizer.convert_tokens_to_ids('[SEP]')
    starttok = tokenizer.convert_tokens_to_ids('<s>')
    endtok = tokenizer.convert_tokens_to_ids('</s>')
    
    model_config_custom = {
                        "bos_token_id": starttok,
                        "eos_token_id": endtok,
                        "decoder_start_token_id": septok,
                        "pad_token_id": padtok,
                        "forced_bos_token_id": None,
                        "no_repeat_ngram_size": 0,
                        "min_length": 0
                         }

    model.config.update(model_config_custom)

    # self.config.length_penalty

    model.eval()    

    ort_sess = None
    bart_script_model = torch.jit.script(GPT2BeamSearchSampleGenerator(model))

    with torch.no_grad():
        PROMT = ["старик", "Кощей"]

        context = " _kw_ ".join(PROMT)
        context = tokenizer.encode(context)
        endkeytok = tokenizer.convert_tokens_to_ids('_endkw_')

        context = [starttok] + context + [endkeytok] + [septok]
        context = tokenizer.decode(context)

        inputs = tokenizer(context, max_length=model.config.n_ctx, return_tensors="pt").to(model.device)
        
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True,
            decoder_start_token_id=septok,
            pad_token_id=padtok,
            bos_token_id=starttok,
            eos_token_id=endtok,
            forced_eos_token_id=endtok,
            temperature=temperature,
            top_p=top_p
        )

        torch.onnx.export(
            bart_script_model,
            (
                inputs["input_ids"],
                inputs["attention_mask"],
                num_beams,
                max_length,
                temperature,
                top_p
            ),
            onnx_file_path,
            opset_version=14, 
            input_names=["input_ids", "attention_mask", "num_beams", "max_length", "temperature", "top_p"], 
            output_names=["output_ids"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "output_ids": {0: "batch", 1: "seq_out"},
            },
            example_outputs=summary_ids,
            verbose=False
        )

        

        options = onnxruntime.SessionOptions()

        ort_sess = onnxruntime.InferenceSession(onnx_file_path, sess_options=options )
        
        ort_inputs = {
                "input_ids": inputs["input_ids"].cpu().numpy(),
                "attention_mask": inputs["attention_mask"].cpu().numpy(),
                "num_beams": np.array(num_beams, dtype=np.int64),
                "max_length": np.array(max_length, dtype=np.int64),
                "temperature": np.array(temperature, dtype=np.float64),
                "top_p": np.array(top_p, dtype=np.float64),
            }
        
        ort_out = ort_sess.run(
            None,
            ort_inputs
        )


        logger.info(f"Pytorch output: {summary_ids.cpu().numpy()}")
        logger.info(f"Onnx output: {ort_out[0]}")

        logger.info(f"Model exported to {onnx_file_path}, size {round(os.path.getsize(onnx_file_path) / 1024 / 1024, 2)} mb")
        logger.info("Success.")


def main():
    args = parse_args()    
    
    max_length = 5
    num_beams = 4

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity_error()

    device = torch.device(args.device)

    model, tokenizer = load_model_tokenizer(args.model_checkpoint_path, args.model_tokenizer_path, device) 

    model.to(device)

    if args.max_length:
        max_length = args.max_length

    if args.num_beams:
        num_beams = args.num_beams
    
    if args.output_file_path:
        output_name = args.output_file_path
    else:
        output_name = "gpt3.onnx"

    temperature = args.temperature
    top_p = args.top_p

    logger.info("Exporting model to ONNX")
    export_and_validate_model(model, tokenizer, output_name, num_beams, max_length, temperature, top_p)


if __name__ == "__main__":
    main()
