import argparse
import os

import torch
from transformers import AutoModel, AutoTokenizer


def str2bool(value: str) -> bool:
	return str(value).lower() in {"1", "true", "t", "yes", "y"}


def parse_args():
	parser = argparse.ArgumentParser(description="DeepSeek-OCR2 HF image inference")
	parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-OCR-2")
	parser.add_argument("--image_file", type=str, required=True, help="Input image path")
	parser.add_argument("--output_path", type=str, required=True, help="Output directory")
	parser.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Convert the document to markdown. ")
	parser.add_argument("--base_size", type=int, default=1024)
	parser.add_argument("--image_size", type=int, default=768)
	parser.add_argument("--crop_mode", type=str2bool, default=True)
	parser.add_argument("--save_results", type=str2bool, default=True)
	parser.add_argument("--cuda_visible_devices", type=str, default="0")
	return parser.parse_args()


def main():
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

	tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
	model_kwargs = {
		"trust_remote_code": True,
		"use_safetensors": True,
	}
	if torch.cuda.is_available():
		model_kwargs["_attn_implementation"] = "flash_attention_2"

	try:
		model = AutoModel.from_pretrained(args.model_name, **model_kwargs)
	except Exception:
		model_kwargs.pop("_attn_implementation", None)
		model = AutoModel.from_pretrained(args.model_name, **model_kwargs)

	model = model.eval()
	if torch.cuda.is_available():
		model = model.cuda().to(torch.bfloat16)

	res = model.infer(
		tokenizer,
		prompt=args.prompt,
		image_file=args.image_file,
		output_path=args.output_path,
		base_size=args.base_size,
		image_size=args.image_size,
		crop_mode=args.crop_mode,
		save_results=args.save_results,
	)


if __name__ == "__main__":
	main()