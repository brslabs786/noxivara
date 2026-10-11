from PIL import Image, ImageDraw
import os

# Create a colorful geometric image
img = Image.new('RGB', (256, 256), color='white')
draw = ImageDraw.Draw(img)

# Draw some shapes
draw.rectangle([50, 50, 200, 200], fill='red', outline='black', width=5)
draw.ellipse([80, 80, 170, 170], fill='blue', outline='yellow', width=5)
draw.line([0, 0, 256, 256], fill='green', width=10)
draw.line([0, 256, 256, 0], fill='green', width=10)

os.makedirs('data/val_images', exist_ok=True)
img.save('data/val_images/validation_shapes.jpg')
print("Generated synthetic validation image!")
