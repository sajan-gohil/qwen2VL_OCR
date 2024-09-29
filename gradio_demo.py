import gradio as gr
import spaces
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import re
import pytesseract


model =  Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@spaces.GPU
def run_example(image, model_id="Qwen/Qwen2-VL-7B-Instruct"):
    model = model.eval()
    processor = processors[model_id]
    system_prompt = "You are a helpfull assistant to read text in images. Read the text in the image verbatim."
    messages = [

        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_to_base64(image)}"},
                {"type": "text", "text": system_prompt},
                # {"type": "text", "text": text_input},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text)
    return output_text

css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

# Function to extract text from the image
def extract_text_from_image(image_path):
    # Convert image to text using pytesseract
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

# Function to highlight specified text
def highlight_text(input_text, keywords):
    # Split keywords by comma and strip whitespace
    keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    
    # Sort keywords by length to prevent conflicts during replacement
    keyword_list.sort(key=len, reverse=True)

    # Initialize an empty list to store highlighted sections
    highlighted = []
    
    # Start index tracking
    last_idx = 0
    
    # Use regular expression to find all occurrences of each keyword
    pattern = "|".join(re.escape(kw) for kw in keyword_list)
    
    # Iterate over all matches and split the text accordingly
    for match in re.finditer(pattern, input_text):
        start_idx, end_idx = match.span()
        
        # Append the text before the match as normal
        if last_idx < start_idx:
            highlighted.append((input_text[last_idx:start_idx], None))  # No label
        
        # Append the matched keyword as highlighted (no label)
        highlighted.append((input_text[start_idx:end_idx], ""))  # No label
        
        # Update the last index
        last_idx = end_idx
    
    # Append the remaining part of the text
    if last_idx < len(input_text):
        highlighted.append((input_text[last_idx:], None))  # No label
    
    return highlighted

# Create Gradio interface
def create_interface():
    # Create the interface layout
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="Upload an Image")

            with gr.Column():
                extracted_text_output = gr.Textbox(label="Extracted Text")
                highlight_phrase = gr.Textbox(label="Enter comma separated phrases to highlight")
                button = gr.Button("Submit")
                highlighted_text_output = gr.HighlightedText(label=None, show_legend=False, interactive=False, show_inline_category=False)

        # Define what happens when an image is uploaded
        image_input.change(fn=run_example, inputs=image_input, outputs=extracted_text_output)
        # Define what happens when text and phrase are inputted
        button.click(fn=highlight_text, inputs=[extracted_text_output, highlight_phrase], outputs=highlighted_text_output)

    return demo


# Run the interface
app = create_interface()
app.launch(share=True)
