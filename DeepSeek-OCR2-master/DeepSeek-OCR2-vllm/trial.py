import fitz
from PIL import Image
import io
import argparse
import os

def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    pdf2images
    """
    images = []

    if not os.path.exists(pdf_path):
        print("pdf path non exist")
    pdf_document = fitz.open(pdf_path)
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
        
        images.append(img)
    
    pdf_document.close()
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OCR pipeline on PDF input."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input PDF file"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory for output files"
    )

    args = parser.parse_args()

    # Override variables from args
    INPUT_PATH = args.input_path
    OUTPUT_PATH = args.output_path

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)
    
    images = pdf_to_images_high_quality(INPUT_PATH)
