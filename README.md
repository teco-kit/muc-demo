# PictographAI

PictographAI is a presentation assistance tool to processes text, images and raw data from presentation slides to automatically generate contextually appropriate pictographs. Our pipeline uses an LLM agent, a text-guided image-inpainting
model, and algorithmic post-processing to make sense of the slide contents and generate meaningful, relevant pictographs

This repository contains a Flask web application of PictographAI that utilizes various machine learning and data processing libraries to perform image inpainting using the `diffusers` library. 
The application also integrates with OpenAI's language models and provides various functionalities such as image processing and data visualization.

## Features

- **Image Inpainting**: Uses `diffusers` library for inpainting images.
- **Data Visualization**: Utilizes `matplotlib` for plotting.
- **Language Model Integration**: Integrates with OpenAI's language models.
- **File Handling**: Supports file uploads and downloads.

## Requirements

- Python 3.8+
- Flask
- torch with CUDA enabled
- diffusers
- pandas
- matplotlib
- Pillow
- langchain_community
- langchain_core

## Installation

1. **Clone the repository**:

    ```sh
    git clone https://github.com/teco-kit/muc-demo/tree/sarah
    cd muc-demo
    ```

2. **Create a virtual environment**:

    ```sh
    python -m venv env
    ```

3. **Activate the virtual environment**:

    ```sh
    .\env\Scripts\activate  # On Windows
    # or
    source env/bin/activate  # On macOS/Linux
    ```

4. **Install the dependencies**:

    ```sh
    pip install -r requirements.txt
    ```

5. **Install the correct PyTorch version**:

    Make sure to install the appropriate PyTorch version for your CUDA driver (if available): [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

6. **Get an OpenAI API key**:

    Get your OpenAI API key and add it to the `App.py` file from here: [OpenAI API Keys](https://platform.openai.com/api-keys)



## Usage

1. **Run the Flask app**:

    ```sh
    python app.py
    ```

2. **Access the application**:

    Open your web browser and go to `http://127.0.0.1:5000`.

## File Structure

- `app.py`: Main application file.
- `templates/`: Directory containing HTML templates.
- `static/`: Directory containing static files (CSS, JavaScript, images).
- `requirements.txt`: List of dependencies.

