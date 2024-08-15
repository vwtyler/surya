import io
import pypdfium2
import streamlit as st
from surya.detection import batch_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.postprocessing.heatmap import draw_polys_on_image
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from PIL import Image
from surya.languages import CODE_TO_LANGUAGE
from surya.input.langs import replace_lang_with_code
from surya.schema import OCRResult, DetectionResult


@st.cache_resource()
def load_det_cached():
    return load_model(), load_processor()


@st.cache_resource()
def load_rec_cached():
    return load_rec_model(), load_rec_processor()


def text_detection(img) -> DetectionResult:
    pred = batch_detection([img], det_model, det_processor)[0]
    polygons = [p.polygon for p in pred.bboxes]
    det_img = draw_polys_on_image(polygons, img.copy())
    return det_img, pred


# Function for OCR
def ocr(img, langs) -> OCRResult:
    replace_lang_with_code(langs)
    img_pred = run_ocr([img], [langs], det_model, det_processor, rec_model, rec_processor)[0]

    bboxes = [l.bbox for l in img_pred.text_lines]
    text = [l.text for l in img_pred.text_lines]
    rec_img = draw_text_on_image(bboxes, text, img.size)
    return rec_img, img_pred


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


@st.cache_data()
def get_all_page_images(pdf_file, dpi=96):
    doc = open_pdf(pdf_file)
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


st.set_page_config(layout="wide")
col1, col2 = st.columns([.5, .5])

det_model, det_processor = load_det_cached()
rec_model, rec_processor = load_rec_cached()

st.markdown("""
# Surya OCR Demo

This app will let you try surya, a multilingual OCR model. It supports text detection in any language, and text recognition in 90+ languages.

Notes:
- This works best on documents with printed text.
- Preprocessing the image (e.g. increasing contrast) can improve results.
- If OCR doesn't work, try changing the resolution of your image (increase if below 2048px width, otherwise decrease).
- This supports 90+ languages, see [here](https://github.com/VikParuchuri/surya/tree/master/surya/languages.py) for a full list.

Find the project [here](https://github.com/VikParuchuri/surya).
""")

in_file = st.sidebar.file_uploader("PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"])
languages = st.sidebar.multiselect("Languages", sorted(list(CODE_TO_LANGUAGE.values())), default=["English"], max_selections=4)

if in_file is None:
    st.stop()

filetype = in_file.type
if "pdf" in filetype:
    pil_images = get_all_page_images(in_file)
else:
    pil_images = [Image.open(in_file).convert("RGB")]

text_det = st.sidebar.button("Run Text Detection")
text_rec = st.sidebar.button("Run OCR")

# Run Text Detection for all pages
if text_det and pil_images:
    for i, pil_image in enumerate(pil_images, start=1):
        det_img, pred = text_detection(pil_image)
        with col1:
            st.image(det_img, caption=f"Detected Text on Page {i}", use_column_width=True)
            st.json(pred.model_dump(exclude=["heatmap", "affinity_map"]), expanded=True)

# Run OCR for all pages
if text_rec and pil_images:
    all_text = []
    for i, pil_image in enumerate(pil_images, start=1):
        rec_img, pred = ocr(pil_image, languages)
        with col1:
            json_tab, text_tab = st.tabs(["JSON", f"Text Lines on Page {i}"])
            with json_tab:
                st.json(pred.model_dump(), expanded=True)
            with text_tab:
                text_content = "\n".join([p.text for p in pred.text_lines])
                st.text(text_content)
                all_text.append(text_content)
    
    # Allow downloading all OCR text as a single file
    full_text_content = "\n\n".join(all_text)
    st.download_button(
        label="Download Full OCR Text",
        data=full_text_content,
        file_name="full_ocr_text.txt",
        mime="text/plain"
    )

with col2:
    st.image(pil_images[0], caption="First Page of Uploaded Document", use_column_width=True)
