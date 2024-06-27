from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import torch
from diffusers import StableDiffusionInpaintPipeline
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
import io
import uuid
import os

import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load your model
device = "cuda"
model_path = "stabilityai/stable-diffusion-2-inpainting"

pipe_2 = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to(device)

key = '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'
prompt = PromptTemplate.from_template(
        "provide me with answer of one word about a relevant object to {content}. The object has to be bar shaped and easily recognizable. Like soda cans, food items, buildings, cigarette packs"
    )

model = OpenAI(openai_api_key = key)
chain = prompt | model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chart', methods=['POST'])

def Chart():

    data = request.get_json()
    title = data.get('title')
    bullet_content = data.get('bulletContent')
    csv_data = data.get('csvData')
    print (csv_data)
    # Decode CSV data

    # Convert the CSV string to a DataFrame
    csv_file = io.StringIO(csv_data)
    df = pd.read_csv(csv_file, header=None)

    # Assuming the CSV has columns 'name' and 'number'

    var_1 = 'Name'
    var_2 = 'Number'
    df.columns = [var_1, var_2]

    # Create a figure and an axes object
    fig, ax = plt.subplots(figsize=(5.12, 5.12))  # Adjusted figure size for better fit

    # Plotting using the axes object
    ax.bar(df[var_1], df[var_2])
    #ax.set_ylim(0, 100)  # Set y-axis limits

    # Add labels using the axes object
    ax.set_xlabel(var_1)
    ax.set_ylabel(var_2)
    ax.set_title(title)


    # Generate a unique ID for the image
    image_id = str(uuid.uuid4())
    image_path = os.path.join('static', f'{image_id}.png')
    mask_image_path = os.path.join('static', f'{image_id}_mask.png')

    # Save the figure to the static directory
    fig.savefig(image_path)
    fig.gca().axes.get_yaxis().set_visible(False)
    fig.gca().axes.get_xaxis().set_visible(False)

    # Remove spines
    fig.gca().spines['top'].set_visible(False)
    fig.gca().spines['right'].set_visible(False)
    fig.gca().spines['bottom'].set_visible(False)
    fig.gca().spines['left'].set_visible(False)

    # Remove title
    fig.gca().title.set_visible(False)
    fig.savefig(mask_image_path)
    # Close the figure to free memory
    plt.close(fig)

    return jsonify({'image_id': image_id})

@app.route('/get_image/<image_id>')
def get_image(image_id):

    image_path = os.path.join('static', f'{image_id}.png')
    
    print ((image_path))
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return "Image not found", 404

@app.route('/picto', methods=['POST'])

def Picto():

    data = request.get_json()
    title = data.get('title')
    bullet_content = data.get('bulletContent')
    csv_data = data.get('csvData')
    originalImageId = data.get('originalImageId')
    print (csv_data)
    # Decode CSV data

    # Convert the CSV string to a DataFrame
    csv_file = io.StringIO(csv_data)
    df = pd.read_csv(csv_file, header=None)

    # Assuming the CSV has columns 'name' and 'number'

    var_1 = 'Name'
    var_2 = 'Number'
    df.columns = [var_1, var_2 ]

    # Create a figure and an axes object
    fig, ax = plt.subplots(figsize=(5.13, 5.13))  # Adjusted figure size for better fit

    # Plotting using the axes object
    ax.bar(df[var_1], df[var_2])
    #ax.set_ylim(0, 100)  # Set y-axis limits

    # Add labels using the axes object
    ax.set_xlabel(var_1)
    ax.set_ylabel(var_2)
    ax.set_title(title)


    fig.savefig('chart.png')

    fig.gca().axes.get_yaxis().set_visible(False)
    fig.gca().axes.get_xaxis().set_visible(False)

    # Remove spines
    fig.gca().spines['top'].set_visible(False)
    fig.gca().spines['right'].set_visible(False)
    fig.gca().spines['bottom'].set_visible(False)
    fig.gca().spines['left'].set_visible(False)

    # Remove title
    fig.gca().title.set_visible(False)
    fig.savefig('mask_chart.png')

    
    ax.bar(df[var_1], df[var_2], color='white')  # Adjusted marker size for better visibility

    fig.gca().axes.get_yaxis().set_visible(True)
    fig.gca().axes.get_xaxis().set_visible(True)
    fig.gca().spines['top'].set_visible(True) 
    fig.gca().spines['right'].set_visible(True)
    fig.gca().spines['bottom'].set_visible(True)
    fig.gca().spines['left'].set_visible(True)
    fig.gca().title.set_visible(True)

    ax.set_xlabel(var_1)
    ax.set_ylabel(var_2)
    ax.set_title(title)

    fig.savefig('axes_chart.png')


    image =  Image.open("chart.png") 


    def create_rectangular_mask(input_image_path, radius=0):
    # Load the image
        image = Image.open(input_image_path)
        image = image.convert("RGBA")  # Ensure image is in RGBA format to check for white pixels

        # Create a blank mask image with the same dimensions as the input image
        mask = Image.new('L', image.size, color=0)  # 'L' mode for black and white (luminance)
        draw = ImageDraw.Draw(mask)

        # Iterate over each pixel in the image
        for x in range(image.width):
            for y in range(image.height):
                r, g, b, a = image.getpixel((x, y))
                # Check if the pixel is not white; consider it non-white if any of the RGB values are not 255
                if r != 255 or g != 255 or b != 255:
                    # Draw a circle of the specified radius around the non-white pixel
                    draw.rectangle((x-radius, y-radius, x+radius, y+radius), fill=255)

        return mask
    # Example usage
    input_image_path = 'mask_chart.png'
    mask_image = create_rectangular_mask(input_image_path,0)

    plt.imsave('grid_mask_image.png', mask_image, cmap='gray') 

    StablePrompt = chain.invoke({'content': title + bullet_content}) 
    print(StablePrompt)

    prompt=StablePrompt[2:] 
    negative_prompt="text, deformed, indistinct"


    image = Image.open('mask_chart.png')
    # Open a mask image file
    mask_image = Image.open("grid_mask_image.png")
    correct_size = (512, 512)
    image = image.resize(correct_size).convert('RGB')
    mask_image = mask_image.resize(correct_size).convert('RGB')


    guidance_scale=20
    num_samples = 1

    generator = torch.Generator(device="cuda").manual_seed(-1) # change the seed to get different results

    images = pipe_2(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
        strength=0.8,
    ).images

    mask_image = mask_image.convert('L')

    transparent_background = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # Apply the mask to the image
    result = Image.composite(images[0], transparent_background, mask_image)

    base_image = Image.open('axes_chart.png').convert('RGBA')
    top_image = result

    # Resize the top image to fit the base image if necessary
    if base_image.size != top_image.size:
        top_image = top_image.resize(base_image.size)

    # Overlay the top image onto the base image
    final_image = Image.alpha_composite(base_image, top_image)

    

    image_id = str(uuid.uuid4())
    image_path = os.path.join('static', f'{image_id}.png')
    final_image.save(image_path)


        
    return jsonify({'image_id': image_id})






if __name__ == '__main__':
    app.run(debug=True)