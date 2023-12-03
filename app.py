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
            entity = str(entity)
            entities.append(entity)

    entities = "\n".join(entities)

    print("ENTITIES ", entities)
    return entities


def run(image, lang="eng"):
    result = pytesseract.image_to_string(image, lang=None if lang == [] else lang)

    ner = get_named_entities(result)
    return result, ner


def download_output(ocr_text: str, named_entities: str):
    print("Download output!")

    print("OCR text: ", len(ocr_text))
    print("Named Entities: ", len(named_entities))

    return True


with gr.Blocks() as demo:
    gr.Markdown("## Theatre Programmer")
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil")
            lang = gr.Dropdown(choices, value="eng")
            btn = gr.Button("Run")
        with gr.Column():
            ocr_text = gr.TextArea(label="OCR output")
        with gr.Column():
            ner = gr.TextArea(label="Named entities")
        # with gr.Column():
        #     gr.CheckboxGroup(ner, label="Named entities")
    with gr.Row():
        download_btn = gr.Button("Download output")

    btn.click(fn=run, inputs=[image_in, lang], outputs=[ocr_text, ner])
    download_btn.click(fn=download_output, inputs=[ocr_text, ner], outputs=[])

demo.launch()
