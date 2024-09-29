---
title: QWEN2VL_OCR_demo
app_file: gradio_demo.py
sdk: gradio
sdk_version: 4.44.0
---

# Approach:
This OCR extractor uses QWEN2-VL model from huggingface, specifically "Qwen/Qwen2-VL-7B-Instruct."
The task document mentioned using colpali, however, colpali is suitable for RAG (retrieval augmented generation) which is not needed for OCR.

The uploaded image is passed to the vision language model with a prompt to return all the data. The generated data is displayed in the extracted text panel. When comma separated keywords are entered after extraction, on clicking the "Submit" button, the text with occurrences of keywords highlighted will be displayed in the Highlighted Text panel. This is done by identifying text segments matching keywords using regex.

**NOTE**: The deployment is on free CPU instance provided by huggingface and hence can take a long time (~1hour) and quantization using huggingface is not available for CPU. If needed, I can deploy on personal AWS instance/sagemaker.
The token limit is 1024 tokens.


# Live demo URL:
`https://huggingface.co/spaces/Sajan/QWEN2VL_OCR_demo`


# Local Setup:
Install dependencies in a python environment
`pip3 install -r requirements.txt`
Run the demo
`python3 gradio_demo.py`
Deployment on huggingface spaces
`gradio deploy`

# Sample outputs:
Image: 8cbac8ffd68c24dd87a017ac152301da.jpg

Output: ['Daily Conversations\n@englishwidsarah\n\nआज मैंने घर पर शांतिपूर्ण दिन बिताया।\nToday i spent a peaceful day at home.\n\nवे कड़ी मेहनत कर रहे हैं।\nThey are working hard.\n\nहमें आपकी सहायता की आवश्यकता है।\nWe need your help.\n\nहमें बस इतना ही चाहिए था।\nThat was all we needed.\n\nयह कैसा महसूस होता है।\nHow does it feel.']

Sample output screenshot: 8cbac8ffd68c24dd87a017ac152301da_output.png