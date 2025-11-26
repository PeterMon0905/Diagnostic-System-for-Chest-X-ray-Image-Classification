import os
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas
from PIL import Image, ImageTk
from torchvision import models

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Can't read image.")
    img = apply_clahe(img)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  
    return img_tensor

def load_model(model_path="model.pth"):
    model = models.densenet121(pretrained=False)
    model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier.in_features, len(LABELS)),
        torch.nn.Sigmoid()
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

class DemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Demo")
        self.root.geometry("600x500")

        # Load model
        self.model = load_model()

        self.image_path = None
        self.photo = None

        self.canvas = Canvas(root, width=300, height=300, bg="gray")
        self.canvas.pack(pady=10)

        self.select_button = Button(root, text="Choose image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.predict_button = Button(root, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)

        self.result_label = Label(root, text="Result: ", wraplength=500, justify="left")
        self.result_label.pack(pady=10)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(title="Choose image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if self.image_path:
            img = Image.open(self.image_path).resize((300, 300))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.result_label.config(text="Result: ")

    def predict(self):
        if not self.image_path:
            self.result_label.config(text="Result: Please choose an image!")
            return

        try:
            img_tensor = preprocess_image(self.image_path).to(DEVICE)
            with torch.no_grad():
                outputs = self.model(img_tensor)
            probs = outputs.cpu().numpy()[0]
            
            results = []
            for i, prob in enumerate(probs):
                results.append(f"{LABELS[i]}: {prob*100:.2f}%")
                # if prob >= 0.5:
                #     results.append(f"{LABELS[i]}: {prob*100:.2f}%")
            
            if results:
                result_text = "Result: " + " ".join(results)
            else:
                result_text = "Result: No pathology detected (all probabilities < 50%)."
            
            self.result_label.config(text=result_text)
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()