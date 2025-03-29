from flask import Flask, request, render_template
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload
        if 'image' not in request.files:
            return "No image uploaded", 400
            
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
            
        # Process image
        try:
            # Process image with BLIP
            try:
                image = Image.open(file.stream).convert('RGB')
                
                # Generate caption
                inputs = processor(image, return_tensors="pt")
                out = model.generate(**inputs)
                description = processor.decode(out[0], skip_special_tokens=True)
                
                if not description:
                    description = "Could not generate a description for this image."

                return render_template('result.html', description=description)
            except Exception as e:
                return "Error generating caption. Please try another image.", 500
                
        except Exception as e:
            return "Error processing image. Please try another image.", 500
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
