import gradio as gr
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
    
    filetype = file_info.name.split('.')[-1].lower()
    if filetype == 'pdf':
        pil_image = get_page_image(file_info.name, page_number)
    else:
        pil_image = Image.open(file_info.name).convert("RGB")
    
    if action == "Display Image":
        # Simply display the uploaded image or the selected PDF page
        return pil_image, "Displaying uploaded image.", None    
    elif action == "Run Text Detection":
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

languages = sorted(list(CODE_TO_LANGUAGE.values()))
action = gr.Radio(["Display Image","Run Text Detection", "Run OCR"], label="Action")
page_number = gr.Number(label="Page Number", value=1, visible=lambda action: action == "Display Image" or action == "Run Text Detection" or action == "Run OCR")

iface = gr.Interface(
    fn=handle_input,
    inputs=[gr.File(label="Upload PDF or Image"), page_number, gr.Dropdown(choices=languages, label="Languages", value=["English"], multiselect=True), action],
    outputs=[gr.Image(label="Output Image"), gr.Text(label="Status Message"), gr.File(label="Download OCR Text")],
    examples=[["invoice.png", 1, ["English"], "Display Image"],["LIC-Education.pdf", 1, ["English"], "Run Text Detection"],["Suriya.png", 1, ["English"], "Run OCR"]],
    title="OCR Suriya",
    description="This app lets you try Suriya, a multilingual OCR model supporting text detection in any language and text recognition in 90+ languages."
)
iface.launch(share=True)