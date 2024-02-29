from PIL import Image
import pypdfium2
import json
from surya.detection import batch_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.postprocessing.heatmap import draw_polys_on_image
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from surya.languages import CODE_TO_LANGUAGE
from surya.input.langs import replace_lang_with_code
from surya.schema import OCRResult, DetectionResult
import tempfile  # For handling temporary file creation
import sys

# Assuming the necessary model loading functions are correctly defined
det_model, det_processor = load_model(), load_processor()
rec_model, rec_processor = load_rec_model(), load_rec_processor()

def text_detection(img) -> DetectionResult:
    pred = batch_detection([img], det_model, det_processor)[0]
    polygons = [p.polygon for p in pred.bboxes]
    det_img = draw_polys_on_image(polygons, img.copy())
    return det_img, pred

def ocr(img, langs) -> OCRResult:
    replace_lang_with_code(langs)
    img_pred = run_ocr([img], [langs], det_model, det_processor, rec_model, rec_processor)[0]

    bboxes = [l.bbox for l in img_pred.text_lines]
    text = [l.text for l in img_pred.text_lines]
    rec_img = draw_text_on_image(bboxes, text, img.size)
    return rec_img, img_pred

def open_pdf(pdf_file_path):
    return pypdfium2.PdfDocument(pdf_file_path)

def get_page_image(pdf_file_path, page_num, dpi=96):
    doc = open_pdf(pdf_file_path)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image

def handle_input(file_info, page_number, languages, action):
    if file_info is None:
        return None, "Please upload a file.", None
    
    filetype = file_info.split('.')[-1].lower()
    if filetype == 'pdf':
        pil_image = get_page_image(file_info, page_number)
    else:
        pil_image = Image.open(file_info).convert("RGB")
      
    if action == "Run Text Detection":
        det_img, _ = text_detection(pil_image)
        return det_img, "Text detection completed.", None
    elif action == "Run OCR":
        rec_img, ocr_result = ocr(pil_image, languages)
        ocr_json = json.dumps(ocr_result.model_dump(), ensure_ascii=False)
        text_lines = "\n".join([line.text for line in ocr_result.text_lines])
        # Save the text lines to a file
        txt_file_path = "ocr_text.txt"
        with open(txt_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(text_lines)
        return rec_img, "OCR completed.", txt_file_path
    else:
        return None, "Please select an action.", None


def main(args):
    if len(args) < 5:
        print("Usage: python script.py <file_path> <page_number> <languages> <action>")
        return

    file_path = args[1]
    page_number = int(args[2])
    languages = args[3]
    action = args[4]


    output, message, file_path = handle_input(file_path, page_number, languages, action)
    print(message)
    if file_path:
        print("Text extracted from OCR has been saved to:", file_path)

if __name__ == "__main__":
    main(sys.argv)
