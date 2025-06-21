import gradio as gr
import huggingface_hub
from PIL import Image
from pathlib import Path
import onnxruntime as rt
import numpy as np
import csv
import os
import io
import tempfile
import shutil
import time

MODEL_REPO = 'toynya/Z3D-E621-Convnext'
THRESHOLD = 0.5
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
DESCRIPTION = """
This is a fork of https://huggingface.co/toynya/Z3D-E621-Convnext
I am not affiliated with the model author in anyway, this is just a useful tool I decided to make based on their model.
"""

print("Downloading model...")
model_path = Path(huggingface_hub.snapshot_download(MODEL_REPO, cache_dir="./models"))
print("Loading model...")
session = rt.InferenceSession(model_path / 'model.onnx', providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

with open(model_path / 'tags-selected.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    tags = [row['name'].strip() for row in csv_reader]

print("Model and tags loaded successfully.")

def prepare_image(image: Image.Image, target_size: int):
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    if image.mode != "RGB":
        image = image.convert("RGB")

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    image_array = np.asarray(padded_image, dtype=np.float32)
    image_array = image_array[:, :, ::-1]
    return np.expand_dims(image_array, axis=0)

def predict(image: Image.Image):
    image_array = prepare_image(image, 448)
    input_name = 'input_1:0'
    output_name = 'predictions_sigmoid'

    result = session.run([output_name], {input_name: image_array})
    result = result[0][0]

    scores = {tags[i]: float(result[i]) for i in range(len(result))}
    predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
    tag_string = ', '.join(predicted_tags)
    return tag_string, scores

def batch_process_folder(folder_path: str, progress=gr.Progress(track_tqdm=True)):
    if not folder_path or not os.path.isdir(folder_path):
        return "Error: Please provide a valid folder path.", None

    output_base_dir = "batch_outputs"
    os.makedirs(output_base_dir, exist_ok=True)
    run_id = str(int(time.time()))
    output_folder = os.path.join(output_base_dir, f"run_{run_id}")
    os.makedirs(output_folder)

    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(SUPPORTED_EXTENSIONS)])

    if not image_files:
        shutil.rmtree(output_folder)
        return f"No supported image files found in the specified folder.", None

    num_digits = len(str(len(image_files)))

    for i, filename in enumerate(progress.tqdm(image_files, desc="Processing Images")):
        base_name = str(i + 1).zfill(num_digits)

        try:
            original_path = os.path.join(folder_path, filename)
            _, extension = os.path.splitext(filename)

            new_image_path = os.path.join(output_folder, f"{base_name}{extension}")
            new_txt_path = os.path.join(output_folder, f"{base_name}.txt")

            shutil.copy(original_path, new_image_path)

            with Image.open(original_path) as img:
                tag_string, _ = predict(img)

            with open(new_txt_path, 'w', encoding='utf-8') as f:
                f.write(tag_string)

        except Exception as e:
            print(f"Could not process {filename}: {e}")
            error_txt_path = os.path.join(output_folder, f"{base_name}_error.txt")
            with open(error_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Could not process original file: {filename}\nError: {e}")

    if not os.listdir(output_folder):
        shutil.rmtree(output_folder)
        return "No files were processed successfully.", None

    zip_output_path_base = os.path.join(output_base_dir, f"results_{run_id}")
    zip_path = shutil.make_archive(
        base_name=zip_output_path_base,
        format='zip',
        root_dir=output_folder
    )

    shutil.rmtree(output_folder)

    status_message = f"Processing complete. {len(image_files)} images and their tags have been zipped."
    return status_message, gr.File(value=zip_path, label="Download Results (.zip)", interactive=True)

print("Starting Gradio server...")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# JoyTag: Image Tagger\n{DESCRIPTION}")

    with gr.Tab("Single Image"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Source Image", sources=['upload', 'webcam'], type='pil')
                submit_btn = gr.Button("Get Tags", variant="primary")
            with gr.Column():
                tag_string_output = gr.Textbox(label="Tag String", interactive=False)
                label_output = gr.Label(label="Tag Predictions", num_top_classes=100)
        submit_btn.click(fn=predict, inputs=image_input, outputs=[tag_string_output, label_output])

    with gr.Tab("Batch Process from Folder"):
        with gr.Column():
            folder_input = gr.Textbox(label="Enter Folder Path", placeholder="/path/to/your/images or /content/images")
            batch_run_btn = gr.Button("Process Folder & Create Zip", variant="primary")
            status_output = gr.Markdown("Awaiting input...")
            file_output = gr.File(label="Download Results (.zip)", interactive=False)
        batch_run_btn.click(fn=batch_process_folder, inputs=folder_input, outputs=[status_output, file_output])

if __name__ == '__main__':
    demo.launch(share=True)
