import torch
import clip
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import Union, List
import numpy as np

logger = logging.getLogger(__name__)

class ClipService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        logger.info(f"CLIP model loaded on {self.device}")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text input to vector embedding
        """
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            # Normalize the features
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # Convert to numpy and return the first (and only) embedding
            return text_features.cpu().numpy()[0]

    def encode_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Encode image input to vector embedding
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        with torch.no_grad():
            # Preprocess and encode the image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
            # Normalize the features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # Convert to numpy and return the first (and only) embedding
            return image_features.cpu().numpy()[0]

    def encode_batch_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of text inputs to vector embeddings
        """
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            # Normalize the features
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()

    def encode_batch_images(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        """
        Encode a batch of image inputs to vector embeddings
        """
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            processed_images.append(self.preprocess(img))

        image_input = torch.stack(processed_images).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # Normalize the features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()

    def get_image_embedding(self, image_path: str) -> torch.Tensor:
        """Generate CLIP image embedding"""
        try:
            with torch.no_grad():
                image = Image.open(image_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                embedding = self.model.encode_image(image_input)
                return F.normalize(embedding, p=2, dim=-1)[0].cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            raise

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Generate CLIP text embedding"""
        try:
            with torch.no_grad():
                text_tokens = clip.tokenize([text]).to(self.device)
                embedding = self.model.encode_text(text_tokens)
                return F.normalize(embedding, p=2, dim=-1)[0].cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise
