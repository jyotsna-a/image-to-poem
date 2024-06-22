from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/generate_poem', methods=['POST'])
def generate_poem():
    # Retrieve the image file from the request
    image_file = request.files.get('image')
    
    if not image_file:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Open and preprocess the image
        image = Image.open(io.BytesIO(image_file.read()))
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate the caption using the BLIP model
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Generate the poem from the caption
        poem = generate_poem_from_caption(caption)
        
        return jsonify({'caption': caption, 'poem': poem})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_poem_from_caption(caption):
    # Placeholder function to generate a poem from a caption
    # Implement your poem generation logic here using LangChain or another method
    return f"Poem for: {caption}"

if __name__ == '__main__':
    app.run(debug=True)
