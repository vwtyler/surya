from cog import BasePredictor, Input, Path,BaseModel
from PIL import Image
import pypdfium2
import json
from typing import List
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

# Assuming the necessary model loading functions are correctly defined
det_model, det_processor = load_model(), load_processor()
rec_model, rec_processor = load_rec_model(), load_rec_processor()

def text_detection(img) -> DetectionResult:
    pred = batch_detection([img], det_model, det_processor)[0]
    polygons = [p.polygon for p in pred.bboxes]
    det_img = draw_polys_on_image(polygons, img.copy())
    return det_img, pred

def ocr(img, langs) -> OCRResult:
    lan = json.loads(langs)
    replace_lang_with_code(lan)
    img_pred = run_ocr([img], [lan], det_model, det_processor, rec_model, rec_processor)[0]

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
        pil_image = get_page_image("/tmp/"+file_info.name, page_number)
    else:
        pil_image = Image.open("/tmp/"+file_info.name).convert("RGB")
      
    if action == "Run Text Detection":
        det_img, _ = text_detection(pil_image)
        # Create a temporary file and save the PIL image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            det_img.save(temp_file, format="JPEG")
            print(temp_file,temp_file.name)
            det_img_path = temp_file.name
        

        return det_img_path, "Text detection completed.", None
    elif action == "Run OCR":
        languages_json = json.dumps(languages)
        rec_img, ocr_result = ocr(pil_image, languages_json)
        # ocr_json = json.dumps(ocr_result.model_dump(), ensure_ascii=False)
        text_lines = "\n".join([line.text for line in ocr_result.text_lines])
        # Save the text lines to a file
        txt_file_path = "ocr_text.txt"
        # Create a temporary file and save the PIL image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            rec_img.save(temp_file, format="JPEG")
            print(temp_file,temp_file.name)
            rec_img_path = temp_file.name

        with open(txt_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(text_lines)
            
        return rec_img_path, "OCR completed.", txt_file_path

class Output(BaseModel):
    image: Path
    text_file: Path = None # Is this used for both actions? 
    Status: str = None  # Placeholder for OCR output

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Model loading logic is already implemented at the beginning
    
    def predict(
        self,
        image: Path = Input(description="Upload PDF or Image"),
        page_number: int = Input(description="Page Number", default=1),
        # languages: str = Input(description="Languages", default='["English"]'),
        languages_choices: str = Input(description="Languages", choices=sorted(list(CODE_TO_LANGUAGE.values())), default="English"),
        languages_input: str = Input(description="Languages (comma-separated list)", default="English"),
        action: str = Input(description="Action", choices=["Run Text Detection", "Run OCR"], default="Run Text Detection"),
    ) -> Output:
        if languages_choices is None:
            selected_languages = [lang.strip() for lang in languages_input.split(',')]
        else:
            selected_languages = [lang.strip() for lang in languages_choices.split(',')]
        output, message, file_path = handle_input(image, page_number, selected_languages, action)
        # Ensure the output matches Cog's expected return types
        # print("******************************************",type(output),print(file_path))
        if action == "Run Text Detection":
            # Construct Text Detection Output
            result = Output(image=Path(output),Status=message)  # Example, adjust based on your logic

        elif action == "Run OCR":
            # Construct OCR Output
            result = Output(image=Path(output), text_file=Path(file_path), Status=message)  

        return result  # Consistent return of an 'Output' object 
