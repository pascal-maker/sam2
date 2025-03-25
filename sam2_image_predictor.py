import numpy as np
from PIL import Image

class SAM2ImagePredictor:
    def __init__(self):
        self.image = None

    @classmethod
    def from_pretrained(cls, model_name):
        print(f"Loaded SAM2 model: {model_name}")
        return cls()

    def set_image(self, image_array):
        self.image = image_array

    def predict(self, input_prompts=None):
        h, w, _ = self.image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if input_prompts:
            for x, y in input_prompts:
                mask[y-10:y+10, x-10:x+10] = 255  # Simple dummy square masks

        masks = [mask]
        scores = [1.0]
        logits = [mask]
        return masks, scores, logits

    def plot(self):
        # Convert mask to color overlay on original image
        overlay = self.image.copy()
        overlay[..., 0] = np.maximum(overlay[..., 0], 100)
        return overlay
