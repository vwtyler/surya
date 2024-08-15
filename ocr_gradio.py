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

# Load models and processors
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

def get_all_page_images(pdf_file_path, dpi=96):
    doc = open_pdf(pdf_file_path)
    images = []
    for page_index in range(len(doc)):
        renderer = doc.render(
            pypdfium2.PdfBitmap.to_pil,
            page_indices=[page_index],
            scale=dpi / 72,
        )
        png = list(renderer)[0]
        png_image = png.convert("RGB")
        images.append(png_image)
    return images

def handle_input(file_info, languages, action):
    if file_info is None:
        return None, "Please upload a file.", None
    
    filetype = file_info.name.split('.')[-1].lower()
    if filetype == 'pdf':
        pil_images = get_all_page_images(file_info.name)
    else:
        pil_images = [Image.open(file_info.name).convert("RGB")]
    
    all_text = []
    if action == "Display Image":
        # Display the first page or the image
        return pil_images[0], "Displaying uploaded image.", None    
    elif action == "Run Text Detection":
        detection_images = []
        for i, pil_image in enumerate(pil_images):
            det_img, _ = text_detection(pil_image)
            detection_images.append(det_img)
        return detection_images[0], "Text detection completed on all pages.", None
    elif action == "Run OCR":
        ocr_images = []
        for i, pil_image in enumerate(pil_images):
            rec_img, ocr_result = ocr(pil_image, languages)
            ocr_images.append(rec_img)
            text_lines = "\n".join([line.text for line in ocr_result.text_lines])
            all_text.append(text_lines)
        
        # Save the text lines from all pages to a file
        full_text_content = "\n\n".join(all_text)
        txt_file_path = "full_ocr_text.txt"
        with open(txt_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(full_text_content)
        return ocr_images[0], "OCR completed on all pages.", txt_file_path
    else:
        return None, "Please select an action.", None

languages = sorted(list(CODE_TO_LANGUAGE.values()))
action = gr.Radio(["Display Image", "Run Text Detection", "Run OCR"], label="Action")

iface = gr.Interface(
    fn=handle_input,
    inputs=[gr.File(label="Upload PDF or Image"), gr.Dropdown(choices=languages, label="Languages", value=["English"], multiselect=True), action],
    outputs=[gr.Image(label="Output Image"), gr.Text(label="Status Message"), gr.File(label="Download OCR Text")],
    examples=[["invoice.png", ["English"], "Display Image"], ["LIC-Education.pdf", ["English"], "Run Text Detection"], ["Suriya.png", ["English"], "Run OCR"]],
    title="OCR Suriya",
    description="This app lets you try Suriya, a multilingual OCR model supporting text detection in any language and text recognition in 90+ languages."
)

iface.launch(share=True)
