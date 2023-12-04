from PIL import Image
import pytesseract
import gradio as gr
from datetime import datetime
import os
from flair.data import Sentence
from flair.models import SequenceTagger
from segtok.segmenter import split_single
import pandas as pd

# tagger = SequenceTagger.load("ner-ontonotes")
tagger = SequenceTagger.load("flair/ner-english-ontonotes")

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

    return entities


def run(image, lang="eng"):
    result = pytesseract.image_to_string(image, lang=None if lang == [] else lang)

    ner = get_named_entities(result)
    return result, ner


def download_output(ocr_text: str, named_entities: str, image_name="test"):
    try:
        named_entities_list = named_entities.split("\n")

        now = datetime.now()
        datetime_now = now.strftime("%Y%m%d_%H%M%S")
        output_file = f"{image_name}_{datetime_now}.xlsx"

        ocr_df = pd.Series(ocr_text)
        print("OCR ", ocr_df)
        ner_df = pd.Series(named_entities_list)
        print("NER ", ner_df)

        with pd.ExcelWriter(output_file) as writer:
            ocr_df.to_excel(writer, sheet_name="OCR text")
            ner_df.to_excel(writer, sheet_name="Named entities")
        return output_file

    except Exception as e:
        raise gr.Error(f"Something went wrong: here's the error: {e}")


with gr.Blocks() as demo:
    gr.Markdown("## Theatre Programmer")
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil")
            lang = gr.Dropdown(choices, value="eng")
            btn = gr.Button("Run")
            print("image_in", image_in.name)
            print("image_in type", type(image_in))
        with gr.Column():
            ocr_text = gr.TextArea(label="OCR output")
        with gr.Column():
            ner = gr.TextArea(label="Named entities")
        # with gr.Column():
        #     gr.CheckboxGroup(ner, label="Named entities")
    with gr.Row():
        download_btn = gr.Button("Download output")

    btn.click(fn=run, inputs=[image_in, lang], outputs=[ocr_text, ner])
    download_btn.click(
        fn=download_output,
        inputs=[ocr_text, ner],
        outputs=[gr.components.File()],
    )

demo.launch()
