---
title: Theatre Programmer
emoji: 🍍
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
license: MIT license
---

Check out the configuration reference at [https://huggingface.co/docs/hub/spaces-config-reference](https://huggingface.co/docs/hub/spaces-config-reference)

## Theatre Programmer

Uses Tesseract to perform OCR on any image supplied as an input. The text identified in the image is then through a Flair Named Entity Recognition (NER) model, the output of which is returned to the page.

This is a prototype produced as part of an SLV LAB project at the State Library Victoria in Melbourne, Australia. Read about the experiment here: [https://lab.slv.vic.gov.au/experiments/theatre-programmes-prototype](https://lab.slv.vic.gov.au/experiments/theatre-programmes-prototype)

### Tech stack

Built using:

- Python
- Gradio web app framework [https://www.gradio.app/](https://www.gradio.app/)

## Repository management

Hugging Face is used to host the demo app, deployment is made via GitHub actions.
