from PIL import Image
import pytesseract
import gradio as gr
import os
from flair.data import Sentence
from flair.models import SequenceTagger
from segtok.segmenter import split_single

tagger = SequenceTagger.load("ner-ontonotes")

langs = []

choices = os.popen("tesseract --list-langs").read().split("\n")[1:-1]

blocks = gr.Blocks()


def get_named_entities(ocr_text: str):
    sentence = [Sentence(sent, use_tokenizer=True) for sent in split_single(ocr_text)]
    tagger.predict(sentence)

    entities = []

    for token in sentence:
        for entity in token.get_spans("ner"):
            entities.append(entity)

    print("ENTITIES ", entities)
    entities = "/n".join(entities)

    return entities


# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string
# print(pytesseract.image_to_string(Image.open('eurotext.png')))

# # French text image to string
# print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))

# # Get bounding box estimates
# print(pytesseract.image_to_boxes(Image.open('test.png')))

# # Get verbose data including boxes, confidences, line and page numbers
# print(pytesseract.image_to_data(Image.open('test.png')))

# # Get information about orientation and script detection
# print(pytesseract.image_to_osd(Image.open('test.png'))


def run(image, lang="eng"):
    result = pytesseract.image_to_string(image, lang=None if lang == [] else lang)

    ner = get_named_entities(result)
    return result, ner


with gr.Blocks() as demo:
    gr.Markdown("## Theatre Programmer")
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil")
            lang = gr.Dropdown(choices, value="eng")
            btn = gr.Button("Run")
        with gr.Column():
            text_out = gr.TextArea(label="OCR output")
        with gr.Column():
            ner = gr.TextArea(label="Named entities")

    btn.click(fn=run, inputs=[image_in, lang], outputs=[text_out, ner])

demo.launch()
