---
title: Theatre Programmer
emoji: 🍍
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Theatre Programmer

Uses Tesseract to perform OCR on any image supplied as an input. The text identified in the image is then through a Flair Named Entity Recognition (NER) model, the output of which is returned to the page.

This is a prototype produced for a project at the State Library Victoria in Melbourne, Australia.

## Repository management

Hugging Face is used to host two version of the demo app:

1. the production version that is linked to the `main` branch of the GitHub repo
2. a development version which is linked to the most recent branch pushed to GitHub that is **not** `main`

Deployment to Hugging Face is done via GitHub actions.
