import gradio as gr
import huggingface_hub
from PIL import Image
from pathlib import Path
import onnxruntime as rt
import numpy as np
import csv


MODEL_REPO = 'toynya/Z3D-E621-Convnext'
THRESHOLD = 0.5
DESCRIPTION = """
This is a demo of https://huggingface.co/toynya/Z3D-E621-Convnext
I am not affiliated with the model author in anyway, this is just a useful tool requested by a user.
"""


def prepare_image(image: Image.Image, target_size: int):
	# Pad image to square
	image_shape = image.size
	max_dim = max(image_shape)
	pad_left = (max_dim - image_shape[0]) // 2
	pad_top = (max_dim - image_shape[1]) // 2

	padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
	padded_image.paste(image, (pad_left, pad_top))

	# Resize
	if max_dim != target_size:
		padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
	
	# Convert to numpy array
	# Based on the ONNX graph, the model appears to expect inputs in the range of 0-255
	image_array = np.asarray(padded_image, dtype=np.float32)

	# Convert PIL-native RGB to BGR
	image_array = image_array[:, :, ::-1]

	return np.expand_dims(image_array, axis=0)


def predict(image: Image.Image):
	image_array = prepare_image(image, 448)

	image_array = prepare_image(image, 448)
	input_name = 'input_1:0'
	output_name = 'predictions_sigmoid'

	result = session.run([output_name], {input_name: image_array})
	result = result[0][0]

	scores = {tags[i]: result[i] for i in range(len(result))}
	predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
	tag_string = ', '.join(predicted_tags)

	return tag_string, scores


print("Downloading model...")
path = Path(huggingface_hub.snapshot_download(MODEL_REPO))
print("Loading model...")
session = rt.InferenceSession(path / 'model.onnx', providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

with open(path / 'tags-selected.csv', mode='r', encoding='utf-8') as file:
	csv_reader = csv.DictReader(file)
	tags = [row['name'].strip() for row in csv_reader]

print("Starting server...")

gradio_app = gr.Interface(
	predict,
	inputs=gr.Image(label="Source", sources=['upload', 'webcam'], type='pil'),
	outputs=[
		gr.Textbox(label="Tag String"),
		gr.Label(label="Tag Predictions", num_top_classes=100),
	],
	title="JoyTag",
	description=DESCRIPTION,
	allow_flagging="never",
)


if __name__ == '__main__':
	gradio_app.launch()
