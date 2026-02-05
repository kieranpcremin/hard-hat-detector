"""
inference.py - Make Predictions on New Images

This module handles loading a trained model and making predictions
on individual images.

KEY CONCEPTS:
-------------
- Model Loading: Restoring saved weights from a checkpoint file
- Inference Mode: Running the model without gradient tracking (faster, less memory)
- Softmax: Converting raw model outputs (logits) to probabilities
"""

from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from model import create_model, get_device


class HardHatPredictor:
    """
    Wrapper class for making predictions with the trained model.

    Usage:
        predictor = HardHatPredictor("models/best_model.pth")
        result = predictor.predict("path/to/image.jpg")
        print(result)
        # {'class': 'hard_hat', 'confidence': 0.95, 'probabilities': {...}}
    """

    # Class names matching our dataset
    CLASSES = ['no_hard_hat', 'hard_hat']

    def __init__(self, model_path: str, device: torch.device = None):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path: Path to the saved model checkpoint (.pth file)
            device: Device to run inference on (default: auto-detect)
        """
        self.device = device or get_device()

        # Create model architecture (same as training)
        print("Loading model...")
        self.model = create_model(num_classes=2, pretrained=False, freeze_backbone=False)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Move to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Validation accuracy: {checkpoint.get('val_acc', 0) * 100:.2f}%")

        # Define transforms (same as validation transforms)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path: str) -> Dict:
        """
        Make a prediction on a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing:
                - 'class': Predicted class name ('hard_hat' or 'no_hard_hat')
                - 'confidence': Confidence score (0.0 to 1.0)
                - 'probabilities': Dict of class probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image)

        # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = self.model(input_batch)

            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)

            # Get predicted class and confidence
            confidence, predicted_idx = torch.max(probabilities, 1)

        # Convert to Python types
        predicted_class = self.CLASSES[predicted_idx.item()]
        confidence_score = confidence.item()

        # Build probabilities dictionary
        probs_dict = {
            class_name: probabilities[0][idx].item()
            for idx, class_name in enumerate(self.CLASSES)
        }

        return {
            'class': predicted_class,
            'confidence': confidence_score,
            'probabilities': probs_dict
        }

    def predict_batch(self, image_paths: list) -> list:
        """
        Make predictions on multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                result['image_path'] = str(path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': str(path),
                    'error': str(e)
                })
        return results


def predict_single_image(
    image_path: str,
    model_path: str = "models/best_model.pth"
) -> Dict:
    """
    Convenience function to predict on a single image.

    Args:
        image_path: Path to the image
        model_path: Path to the model checkpoint

    Returns:
        Prediction dictionary
    """
    predictor = HardHatPredictor(model_path)
    return predictor.predict(image_path)


# =============================================================================
# MAIN - Example usage
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Predict hard hat presence in an image")
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to model checkpoint")

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("Please train the model first using train.py")
        sys.exit(1)

    # Make prediction
    result = predict_single_image(args.image_path, args.model)

    # Display result
    print("\n" + "=" * 40)
    print("PREDICTION RESULT")
    print("=" * 40)
    print(f"Image: {args.image_path}")
    print(f"Prediction: {result['class'].upper()}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print("\nProbabilities:")
    for class_name, prob in result['probabilities'].items():
        bar = "█" * int(prob * 20)
        print(f"  {class_name:15} {prob * 100:5.1f}% {bar}")
