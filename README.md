# e621.net Tagger

A simple Gradio GUI for automatically tagging images for LoRa training datasets. My tool is based on [Z3D-E621-Convnext-space](https://huggingface.co/spaces/fancyfeast/Z3D-E621-Convnext-space) to generate descriptive e621 tags for your images.

You can run this application easily on Google Colab or on your local machine.

## ✨ Features

-   **Simple UI**: Clean gradio interface.
-   **Single File Processing**: You can quickly tag a single image by uploading it directly through the gradio app.
-   **Batch Processing**: Process an entire folder of images at once by giving it the path to your dataset.
-   **Automated Output**: Generates a `.txt` file for each image with the corresponding tags.
-   **Organized Results**: Batch processing outputs a convenient `.zip` file with images and their tag files, numerically ordered.
-   **Flexible Usage**: Run it anywhere with Python, or on Google Colab using my notebook.

---

## How to Use

There are two primary ways to run this tagger.

### 1. Google Colab (Recommended)

The easiest way to get started. No installation required. Simple and streamlined for newcomers.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/litsa-the-dancer/bbb64508867cb136a1b64a570ac44a28/e621tagger.ipynb)

### 2. Local Installation

For those who prefer to run it locally:

**Prerequisites:**
*   Python 3.8+
*   `git` (for cloning the repository)

**Setup:**

1.  **Clone the repository:**
    ```bash
    git https://github.com/litsa-the-dancer/e621.net-tagger.git
    cd e621.net-tagger
    ```

2.  **Create and activate a virtual environment (recommended but totally optional):**
    A virtual environment keeps your project's dependencies separate from your system's Python installation.
    ```bash
    # for windows
    python -m venv venv
    .\venv\Scripts\activate

    # for macOS/linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    Once it's running, you'll see a local URL and Gradio URL in your terminal. Open one of those links in your web browser to use the GUI, whichever one you prefer.

---

## How It Works

The GUI is divided into two main sections for different workflows.

### Single File Processing

1.  Navigate to the "Single File" tab.
2.  Upload or drag-and-drop your image into the input box.
3.  Click "Process" and wait for the tags to be generated.

### Batch Processing

This mode is designed for tagging an entire dataset folder.

1.  Navigate to the "Batch Process" tab.
2.  Enter the **full path** to the folder containing your images (e.g., `C:\Users\YourName\Desktop\lora_dataset`).
3.  Click "Process". The tagger will process each image one by one.
4.  Once complete, a zip will be generated. This zip file contains all your original images, each with a corresponding `.txt` file containing its tags. The files are numerically ordered for convenience.

> **Important**: This tool generates descriptive tags only. You will need to **manually add your own activation tag** (trigger word) to each text file before training your LoRa.

---

## Future Plans

I may add more features in the future, such as:
-   Customizable tag filtering or exclusion.
-   Option to automatically add a specified activation tag to all files.

## Disclaimer

I will not be held liable for any actions taken using this repository. The user of this software is solely responsible for how they use it and for any content they generate with it.
