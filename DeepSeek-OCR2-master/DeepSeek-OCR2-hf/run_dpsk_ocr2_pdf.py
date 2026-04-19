import argparse
import io
import os
import re
import tempfile
from pathlib import Path

import fitz
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "t", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR2 PDF inference with Hugging Face Transformers")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-OCR-2")
    parser.add_argument("--input-path", "--input_path", dest="input_path", type=str, required=True, help="Input PDF path")
    parser.add_argument("--output-path", "--output_path", dest="output_path", type=str, required=True, help="Output directory")
    parser.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Convert the document to markdown. ")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--crop-mode", type=str2bool, default=True)
    parser.add_argument("--save-results", type=str2bool, default=True)
    parser.add_argument("--skip-repeat", type=str2bool, default=False)
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()

def pdf_to_images_high_quality(pdf_path, dpi=144):
    images = []
    pdf_document = fitz.open(pdf_path)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        images.append(img)

    pdf_document.close()
    return images


def pil_to_pdf_img2pdf(pil_images, output_path):
    if not pil_images:
        return

    rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in pil_images]

    first_img, *rest_imgs = rgb_images
    first_img.save(output_path, format="PDF", save_all=True, append_images=rest_imgs)


def re_match(text):
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for match in matches:
        if "<|ref|>image<|/ref|>" in match[0]:
            matches_image.append(match[0])
        else:
            matches_other.append(match[0])
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
        return label_type, cor_list
    except Exception:
        return None


def draw_bounding_boxes(image, refs, page_index, image_output_dir):
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    image_idx = 0
    for ref in refs:
        result = extract_coordinates_and_label(ref)
        if not result:
            continue

        label_type, points_list = result
        color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
        color_a = color + (20,)

        for points in points_list:
            try:
                x1, y1, x2, y2 = points
                x1 = int(x1 / 999 * image_width)
                y1 = int(y1 / 999 * image_height)
                x2 = int(x2 / 999 * image_width)
                y2 = int(y2 / 999 * image_height)

                if label_type == "image":
                    cropped = image.crop((x1, y1, x2, y2))
                    cropped.save(os.path.join(image_output_dir, f"{page_index}_{image_idx}.jpg"))
                    image_idx += 1

                if label_type == "title":
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    draw_overlay.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                else:
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    draw_overlay.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                text_x = x1
                text_y = max(0, y1 - 15)
                text_bbox = draw.textbbox((0, 0), label_type, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=(255, 255, 255, 30))
                draw.text((text_x, text_y), label_type, font=font, fill=color)
            except Exception:
                continue

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def get_text_from_infer_result(result):
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ["text", "response", "result", "output", "content"]:
            if key in result and isinstance(result[key], str):
                return result[key]
    if isinstance(result, (list, tuple)) and result:
        first_item = result[0]
        if isinstance(first_item, str):
            return first_item
        if isinstance(first_item, dict):
            for key in ["text", "response", "result", "output", "content"]:
                if key in first_item and isinstance(first_item[key], str):
                    return first_item[key]
    return str(result)


def build_model(model_name, dtype_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_name]

    model_kwargs = {
        "trust_remote_code": True,
        "use_safetensors": True,
    }
    if torch.cuda.is_available():
        model_kwargs["_attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    except Exception:
        model_kwargs.pop("_attn_implementation", None)
        model = AutoModel.from_pretrained(model_name, **model_kwargs)

    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda().to(dtype)

    return tokenizer, model


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    output_dir = Path(args.output_path)
    image_output_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    tokenizer, model = build_model(args.model_name, args.dtype)

    print("Loading PDF pages...")
    images = pdf_to_images_high_quality(args.input_path, dpi=args.dpi)

    base_name = Path(args.input_path).stem
    mmd_det_path = output_dir / f"{base_name}_det.mmd"
    mmd_path = output_dir / f"{base_name}.mmd"
    pdf_out_path = output_dir / f"{base_name}_layouts.pdf"

    contents_det = []
    contents = []
    draw_images = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for page_index, image in enumerate(tqdm(images, desc="Inferencing pages")):
            page_image_path = os.path.join(tmp_dir, f"page_{page_index}.png")
            image.save(page_image_path)

            result = model.infer(
                tokenizer,
                prompt=args.prompt,
                image_file=page_image_path,
                output_path=str(output_dir),
                base_size=args.base_size,
                image_size=args.image_size,
                crop_mode=args.crop_mode,
                save_results=args.save_results,
            )
            

            content = get_text_from_infer_result(result)
            if "<｜end▁of▁sentence｜>" in content:
                content = content.replace("<｜end▁of▁sentence｜>", "")
            elif args.skip_repeat:
                continue

            page_split = "\n<--- Page Split --->"
            contents_det.append(content + f"\n{page_split}\n")

            matches_ref, matches_images, matches_other = re_match(content)
            result_image = draw_bounding_boxes(image.copy(), matches_ref, page_index, str(image_output_dir))
            draw_images.append(result_image)

            for image_idx, match_image in enumerate(matches_images):
                content = content.replace(match_image, f"![](images/{page_index}_{image_idx}.jpg)\n")

            for match_other in matches_other:
                content = (
                    content.replace(match_other, "")
                    .replace("\\coloneqq", ":=")
                    .replace("\\eqqcolon", "=:")
                    .replace("\n\n\n\n", "\n\n")
                    .replace("\n\n\n", "\n\n")
                )

            contents.append(content + f"\n{page_split}\n")

    with open(mmd_det_path, "w", encoding="utf-8") as f_det:
        f_det.write("".join(contents_det))

    with open(mmd_path, "w", encoding="utf-8") as f_clean:
        f_clean.write("".join(contents))

    pil_to_pdf_img2pdf(draw_images, str(pdf_out_path))
    print(f"Saved: {mmd_det_path}")
    print(f"Saved: {mmd_path}")
    print(f"Saved: {pdf_out_path}")


if __name__ == "__main__":
    main()