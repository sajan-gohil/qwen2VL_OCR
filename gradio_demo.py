import gradio as gr
# import spaces
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import re
# from transformers import BitsAndBytesConfig
# bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    # quantization_config=bnb_config
)
# model = torch.quantization.quantize_dynamic(
#     model, 
#     {torch.nn.Linear},  # Quantize linear layers
#     dtype=torch.qint8  # Use 8-bit integer precision
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


def image_to_base64(image):
    print("image = ", image, flush=True)
    image = Image.open(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# @spaces.GPU
def run_example(image):
    """
    Pass image to LLM with OCR prompt
    """
    global model
    global processor
    print(model)
    model = model.eval()
    with torch.no_grad():
        system_prompt = "You are a helpfull assistant to read text in images. Read the text in the image verbatim."
        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image;base64,{image_to_base64(image)}"
                },
                {
                    "type": "text",
                    "text": system_prompt
                },
                # {"type": "text", "text": text_input},
            ],
        }]

        text = processor.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        print(output_text)
        return output_text


def highlight_text(input_text, keywords):
    """
    Function to highlight specified text
    """
    # Preprocesses all keywords with largest first and find iteratively using regex
    keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    keyword_list.sort(key=len, reverse=True)
    highlighted = []
    # Keep track of last found word index, find next match and add everything in between as normal text
    last_idx = 0
    pattern = "|".join(re.escape(kw) for kw in keyword_list)
    # Iterate over all matches and split the text accordingly
    for match in re.finditer(pattern, input_text):
        start_idx, end_idx = match.span()

        # Normal text from end of previous match till start of current (next found)
        if last_idx < start_idx:
            highlighted.append(
                (input_text[last_idx:start_idx], None))

        # Matched text
        highlighted.append((input_text[start_idx:end_idx], ""))
        last_idx = end_idx

    if last_idx < len(input_text):
        highlighted.append((input_text[last_idx:], None))

    return highlighted


def create_interface():
    """
    Gradio Interface
    """
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath",
                                       label="Upload an Image")

            with gr.Column():
                extracted_text_output = gr.Textbox(label="Extracted Text")
                highlight_phrase = gr.Textbox(
                    label="Enter comma separated phrases to highlight")
                button = gr.Button("Submit")
                highlighted_text_output = gr.HighlightedText(
                    label=None,
                    show_legend=False,
                    interactive=False,
                    show_inline_category=False)

        # Run OCR when image is uploaded
        image_input.change(fn=run_example,
                           inputs=image_input,
                           outputs=extracted_text_output)
        # Pass OCR output and keywords to highlighted text processor
        button.click(fn=highlight_text,
                     inputs=[extracted_text_output, highlight_phrase],
                     outputs=highlighted_text_output)

    return demo

if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)
