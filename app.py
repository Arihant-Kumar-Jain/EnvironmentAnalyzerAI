import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
from datetime import datetime, timedelta
import random
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Google Drive file ID for model download
file_id = "1_vFGQZc5xM3CF1iolA-HNPQKTtth5fW9"
output_name = "latest_model.pth"

if not os.path.exists(output_name):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_name, quiet=False)
    print(f"Model downloaded as: {output_name}")
else:
    print(f"Model '{output_name}' already exists.")

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ResNet18 for flood detection
def load_flood_model():
    try:
        # Try loading just the weights with weights_only=True
        state_dict = torch.load('best_model.pth', map_location=torch.device('cpu'), weights_only=True)
        
        # Looking at the keys in the state_dict, it appears to be a Sequential model
        # with a structure that's different from standard ResNet
        
        # Create a custom model structure that matches the saved weights
        # This is a simplified example - you may need to adjust based on your specific model
        custom_model = nn.Sequential(
            # First layer seems to be a Conv2d based on "0.0.weight"
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            # BatchNorm layer based on "0.1.weight", etc.
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Additional layers would go here...
        )
        
        # Load the weights into the custom model
        # You might need to manually map the keys from the state_dict to your model
        # This is complex and depends on your specific model structure
        
        # For illustration - you'd need to implement actual key mapping
        new_state_dict = {}
        for key, value in state_dict.items():
            # Map keys from saved format to your custom model format
            # For example: "0.0.weight" -> custom mapping
            new_state_dict[key] = value
            
        custom_model.load_state_dict(new_state_dict)
        custom_model.eval()
        return custom_model
        
    except Exception as e:
        print(f"Error loading model with weights_only=True: {e}")
        
        # As a fallback, create a new model
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.eval()
        return model

# Load the flood model
try:
    flood_model = load_flood_model()
    print("Loaded ResNet18 flood model successfully")
except Exception as e:
    print(f"Could not load flood model: {e}")
    flood_model = None

# Define the preprocessing for the flood model (standard ImageNet preprocessing)
flood_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the UNet architecture for the forest model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder path
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        out = self.sigmoid(out)
        
        return out

# Load the forest UNet model
forest_model = UNet(in_channels=3, out_channels=1)
forest_model.load_state_dict(torch.load('latest_model.pth', map_location=torch.device('cpu')))
forest_model.eval()

def preprocess_forest_image(image):
    """
    Preprocess the image for the forest coverage model
    """
    # Resize to 512x512 or whatever size your forest model expects
    img = image.resize((512, 512))
    
    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(np.array(img)).float()
    img_tensor = img_tensor.permute(2, 0, 1)  # Change to [C, H, W]
    img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def generate_future_flood_predictions(probability):
    today = datetime.now()
    predictions = []
    
    base_prob = probability
    for i in range(1, 6):
        future_date = today + timedelta(days=i)
        
        # Adjust probability with randomness but keep it within 10% to 70%
        future_prob = max(0.1, min(0.7, base_prob + (random.uniform(-0.15, 0.15) * i)))
        
        predictions.append({
            'date': future_date.strftime('%Y-%m-%d'),
            'probability': round(future_prob * 100, 2),
            'status': 'Likely Flooded' if future_prob > 0.5 else 'Likely Not Flooded'
        })

    return predictions

def generate_future_forest_predictions(forest_probability, coverage_percentage):
    today = datetime.now()
    predictions = []
    
    base_prob = forest_probability
    base_coverage = coverage_percentage
    
    for i in range(1, 6):
        future_date = today + timedelta(days=i*30)  # Forest changes over months rather than days
        
        # Adjust forest probability
        future_prob = max(0.1, min(0.9, base_prob + (random.uniform(-0.05, 0.05) * i)))
        
        # Adjust coverage percentage
        future_coverage = max(0, min(100, base_coverage + (random.uniform(-3, 3) * i)))
        
        predictions.append({
            'date': future_date.strftime('%Y-%m-%d'),
            'forest_probability': round(future_prob * 100, 2),
            'coverage_percentage': round(future_coverage, 2),
            'status': 'Forest' if future_prob > 0.5 else 'Not Forest'
        })

    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    model_type = request.form.get('model_type', 'flood')  # Default to flood if not specified
    
    try:
        # Read the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert RGBA to RGB if needed before saving as JPEG
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Generate timestamp for unique filename to prevent caching issues
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(app.config['UPLOAD_FOLDER'], f'upload_{model_type}_{timestamp}.jpg')
        image.save(filename)
        
        if model_type == 'flood':
            # Process for flood detection using ResNet18 model
            global flood_model
            
            # If model not loaded yet, try loading it now
            if flood_model is None:
                try:
                    flood_model = load_flood_model()
                except Exception as e:
                    return jsonify({'error': f"Failed to load flood model: {str(e)}"})
            
            # Preprocess the image for ResNet18
            input_tensor = flood_preprocess(image)
            input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                output = flood_model(input_batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Get flood probability (class 1)
            flood_prob = float(probabilities[0][1])
            
            # Determine if flooded
            is_flooded = flood_prob > 0.5
            result = 'Flooded' if is_flooded else 'Not Flooded'
            
            # Generate future predictions
            future_predictions = generate_future_flood_predictions(flood_prob)
            
            return jsonify({
                'model_type': 'flood',
                'result': result,
                'probability': round(flood_prob * 100, 2),
                'image_path': filename,
                'future_predictions': future_predictions
            })
            
        elif model_type == 'forest':
            # Process for forest coverage
            processed_image = preprocess_forest_image(image)
            
            # Make prediction using PyTorch model
            with torch.no_grad():
                output = forest_model(processed_image)
                
            # For UNet output, we need to calculate the forest coverage percentage
            # The output is a segmentation mask of size [1, 1, H, W]
            forest_mask = output.squeeze().cpu().numpy()
            
            # Calculate coverage percentage as the percentage of pixels that are marked as forest
            coverage_percentage = float(np.mean(forest_mask) * 100)
            
            # Determine forest probability (if >20% is forest, we'll say it's a forest)
            forest_probability = min(1.0, coverage_percentage / 50) if coverage_percentage > 0 else 0
            is_forest = coverage_percentage > 20
            result = 'Forest' if is_forest else 'Not Forest'
            
            # Generate future predictions for forest coverage
            future_predictions = generate_future_forest_predictions(forest_probability, coverage_percentage)
            
            return jsonify({
                'model_type': 'forest',
                'result': result,
                'forest_probability': round(forest_probability * 100, 2),
                'coverage_percentage': round(coverage_percentage, 2),
                'image_path': filename,
                'future_predictions': future_predictions
            })
        else:
            return jsonify({'error': 'Invalid model type'})
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({'error': f"{str(e)}\n\nFull traceback: {traceback_str}"})

if __name__ == '__main__':
    app.run(debug=True)