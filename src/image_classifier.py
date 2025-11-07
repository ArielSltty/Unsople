# image_classifier.py
"""
🧠 Unsople - AI-Powered Smart Sorting System
📍 Image Classification Module
🎯 Core AI model handling for waste classification
"""

import cv2
import numpy as np
import onnxruntime as ort
import json
import time
from PIL import Image, ImageOps
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

class UnsopleImageClassifier:
    """
    Core image classifier for Unsople waste classification system
    Handles ONNX model loading, preprocessing, and inference
    """
    
    # Waste categories supported by Unsople
    WASTE_CATEGORIES = {
        'plastic': {
            'name': 'Plastic',
            'bin_color': 'yellow',
            'examples': ['bottle', 'container', 'wrapper', 'bag']
        },
        'paper': {
            'name': 'Paper',
            'bin_color': 'blue', 
            'examples': ['newspaper', 'cardboard', 'office_paper', 'box']
        },
        'organic': {
            'name': 'Organic',
            'bin_color': 'green',
            'examples': ['food_waste', 'fruit', 'vegetable', 'compost']
        },
        'metal': {
            'name': 'Metal', 
            'bin_color': 'gray',
            'examples': ['can', 'foil', 'container', 'utensils']
        },
        'glass': {
            'name': 'Glass',
            'bin_color': 'brown',
            'examples': ['bottle', 'jar', 'container', 'broken_glass']
        }
    }
    
    def __init__(self, model_path: str, config_path: str = "config.json"):
        """
        Initialize the image classifier
        
        Args:
            model_path (str): Path to ONNX model file
            config_path (str): Path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.logger = self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_metadata = {}
        
        self._load_model()
        self._validate_model()
        
        self.logger.info("✅ Unsople Image Classifier initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the classifier"""
        logger = logging.getLogger('UnsopleImageClassifier')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _load_config(self) -> Dict:
        """
        Load configuration from JSON file
        
        Returns:
            Dict: Configuration parameters
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"📁 Configuration loaded from {self.config_path}")
            return config
            
        except FileNotFoundError:
            self.logger.warning(f"⚠️ Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Error parsing config file: {e}")
            raise
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration when config file is not available
        
        Returns:
            Dict: Default configuration
        """
        return {
            "model": {
                "input_size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "confidence_threshold": 0.5
            },
            "preprocessing": {
                "auto_orient": True,
                "maintain_aspect_ratio": True,
                "padding_color": [114, 114, 114]
            },
            "inference": {
                "providers": ["CPUExecutionProvider"],
                "session_options": {
                    "intra_op_num_threads": 4,
                    "inter_op_num_threads": 4
                }
            }
        }
    
    def _load_model(self):
        """Load ONNX model and create inference session"""
        try:
            self.logger.info(f"🧠 Loading ONNX model from {self.model_path}")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Create session options for optimization
            session_options = ort.SessionOptions()
            
            # Set thread configuration
            session_options.intra_op_num_threads = self.config['inference']['session_options']['intra_op_num_threads']
            session_options.inter_op_num_threads = self.config['inference']['session_options']['inter_op_num_threads']
            
            # Enable optimizations
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create inference session
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=self.config['inference']['providers']
            )
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Extract model metadata
            self._extract_model_metadata()
            
            self.logger.info(f"✅ Model loaded successfully")
            self.logger.info(f"📊 Input: {self.input_name}, Shape: {self.session.get_inputs()[0].shape}")
            self.logger.info(f"📊 Output: {self.output_name}, Shape: {self.session.get_outputs()[0].shape}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _extract_model_metadata(self):
        """Extract and store model metadata"""
        try:
            input_info = self.session.get_inputs()[0]
            output_info = self.session.get_outputs()[0]
            
            self.model_metadata = {
                'input_name': input_info.name,
                'input_shape': input_info.shape,
                'input_type': input_info.type,
                'output_name': output_info.name,
                'output_shape': output_info.shape,
                'output_type': output_info.type,
                'providers': self.session.get_providers()
            }
            
            self.logger.debug(f"📋 Model metadata: {self.model_metadata}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract model metadata: {e}")
    
    def _validate_model(self):
        """Validate model compatibility and requirements"""
        try:
            # Check input dimensions
            input_shape = self.model_metadata['input_shape']
            if len(input_shape) != 4 or input_shape[1] not in [1, 3]:
                self.logger.warning(f"⚠️ Unexpected input shape: {input_shape}")
            
            # Check if model expects RGB input
            if input_shape[1] == 3:
                self.logger.info("✅ Model expects RGB input (3 channels)")
            elif input_shape[1] == 1:
                self.logger.info("ℹ️ Model expects grayscale input (1 channel)")
            
            # Log model capabilities
            self.logger.info(f"🔧 Available execution providers: {self.model_metadata['providers']}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Model validation incomplete: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model inference
        
        Args:
            image (np.ndarray): Input image in BGR format (OpenCV default)
            
        Returns:
            np.ndarray: Preprocessed image ready for inference
        """
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image for advanced processing
            pil_image = Image.fromarray(image_rgb)
            
            # Auto-orient image based on EXIF data
            if self.config['preprocessing']['auto_orient']:
                pil_image = ImageOps.exif_transpose(pil_image)
            
            # Get target size from config
            target_size = tuple(self.config['model']['input_size'])
            
            # Resize image
            if self.config['preprocessing']['maintain_aspect_ratio']:
                # Resize with aspect ratio maintained and padding
                pil_image = self._resize_with_padding(pil_image, target_size)
            else:
                # Simple resize (may distort aspect ratio)
                pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert back to numpy array
            processed_image = np.array(pil_image, dtype=np.float32)
            
            # Normalize pixel values to [0, 1]
            processed_image /= 255.0
            
            # Apply normalization (ImageNet stats by default)
            mean = np.array(self.config['model']['mean'], dtype=np.float32)
            std = np.array(self.config['model']['std'], dtype=np.float32)
            
            processed_image = (processed_image - mean) / std
            
            # Convert to CHW format (Channel, Height, Width)
            processed_image = np.transpose(processed_image, (2, 0, 1))
            
            # Add batch dimension
            processed_image = np.expand_dims(processed_image, axis=0)
            
            return processed_image
            
        except Exception as e:
            self.logger.error(f"❌ Image preprocessing failed: {e}")
            raise
    
    def _resize_with_padding(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image while maintaining aspect ratio with padding
        
        Args:
            image (Image.Image): Input PIL image
            target_size (tuple): Target (width, height)
            
        Returns:
            Image.Image: Resized image with padding
        """
        target_w, target_h = target_size
        image_w, image_h = image.size
        
        # Calculate scaling factor
        scale = min(target_w / image_w, target_h / image_h)
        
        # Calculate new dimensions
        new_w = int(image_w * scale)
        new_h = int(image_h * scale)
        
        # Resize image
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_image = Image.new(
            'RGB', 
            (target_w, target_h), 
            tuple(self.config['preprocessing']['padding_color'])
        )
        
        # Calculate padding position (center)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        
        # Paste resized image
        new_image.paste(resized_image, (paste_x, paste_y))
        
        return new_image
    
    def postprocess_prediction(self, model_output: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Process model output to get final prediction
        
        Args:
            model_output (np.ndarray): Raw model output
            
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        try:
            # Apply softmax to get probabilities
            probabilities = self._softmax(model_output)
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[0][predicted_idx]
            
            # Get class names from model metadata or use defaults
            class_names = self._get_class_names()
            
            # Check if predicted index is valid
            if predicted_idx >= len(class_names):
                self.logger.warning(f"⚠️ Predicted index {predicted_idx} out of range")
                return None, 0.0, {}
            
            predicted_class = class_names[predicted_idx]
            
            # Create probability dictionary for all classes
            all_probabilities = {
                class_name: float(probabilities[0][i]) 
                for i, class_name in enumerate(class_names)
            }
            
            # Apply confidence threshold
            confidence_threshold = self.config['model']['confidence_threshold']
            if confidence < confidence_threshold:
                self.logger.debug(f"📊 Prediction below threshold: {predicted_class} ({confidence:.3f} < {confidence_threshold})")
                return None, confidence, all_probabilities
            
            self.logger.debug(f"🎯 Prediction: {predicted_class} (confidence: {confidence:.3f})")
            return predicted_class, confidence, all_probabilities
            
        except Exception as e:
            self.logger.error(f"❌ Prediction postprocessing failed: {e}")
            return None, 0.0, {}
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax function to model outputs
        
        Args:
            x (np.ndarray): Model output logits
            
        Returns:
            np.ndarray: Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _get_class_names(self) -> List[str]:
        """
        Get class names from model metadata or configuration
        
        Returns:
            List[str]: List of class names
        """
        # Try to get from model metadata first
        if 'classes' in self.config:
            classes_dict = self.config['classes']
            # Convert to list maintaining index order
            class_names = [None] * len(classes_dict)
            for idx, name in classes_dict.items():
                class_names[int(idx)] = name
            return class_names
        
        # Fallback to default waste categories
        return list(self.WASTE_CATEGORIES.keys())
    
    def predict(self, image: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Perform waste classification on input image
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
                   Returns (None, 0.0, {}) if prediction fails or below threshold
        """
        try:
            start_time = time.time()
            
            # Validate input image
            if image is None or image.size == 0:
                self.logger.error("❌ Invalid input image")
                return None, 0.0, {}
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Run inference
            model_output = self.session.run(
                [self.output_name], 
                {self.input_name: processed_image}
            )[0]
            
            # Postprocess results
            prediction, confidence, all_probabilities = self.postprocess_prediction(model_output)
            
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.logger.debug(f"⏱️ Inference time: {inference_time:.2f}ms")
            
            return prediction, confidence, all_probabilities
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            return None, 0.0, {}
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[Optional[str], float, Dict[str, float]]]:
        """
        Perform batch prediction on multiple images
        
        Args:
            images (List[np.ndarray]): List of input images
            
        Returns:
            List of tuples: Each containing (predicted_class, confidence, all_probabilities)
        """
        results = []
        
        for i, image in enumerate(images):
            self.logger.debug(f"🔍 Processing image {i+1}/{len(images)}")
            result = self.predict(image)
            results.append(result)
        
        return results
    
    def get_class_info(self, class_name: str) -> Optional[Dict]:
        """
        Get detailed information about a waste class
        
        Args:
            class_name (str): Name of the waste class
            
        Returns:
            Dict: Class information or None if not found
        """
        return self.WASTE_CATEGORIES.get(class_name.lower())
    
    def get_bin_recommendation(self, class_name: str) -> str:
        """
        Get bin color recommendation for waste class
        
        Args:
            class_name (str): Name of the waste class
            
        Returns:
            str: Bin color recommendation
        """
        class_info = self.get_class_info(class_name)
        if class_info:
            return class_info['bin_color']
        return 'unknown'
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information
        
        Returns:
            Dict: Model information and capabilities
        """
        return {
            'model_path': self.model_path,
            'config_path': self.config_path,
            'metadata': self.model_metadata,
            'waste_categories': list(self.WASTE_CATEGORIES.keys()),
            'input_requirements': {
                'format': 'BGR (OpenCV default)',
                'size': self.config['model']['input_size'],
                'normalization': 'ImageNet statistics'
            },
            'performance': {
                'confidence_threshold': self.config['model']['confidence_threshold'],
                'providers': self.model_metadata['providers']
            }
        }
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate if image is suitable for classification
        
        Args:
            image (np.ndarray): Input image to validate
            
        Returns:
            bool: True if image is valid
        """
        if image is None:
            self.logger.error("❌ Image is None")
            return False
        
        if image.size == 0:
            self.logger.error("❌ Image is empty")
            return False
        
        if len(image.shape) not in [2, 3]:
            self.logger.error(f"❌ Invalid image shape: {image.shape}")
            return False
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            self.logger.error(f"❌ Invalid number of channels: {image.shape[2]}")
            return False
        
        min_dimension = 32  # Minimum reasonable dimension for classification
        if image.shape[0] < min_dimension or image.shape[1] < min_dimension:
            self.logger.warning(f"⚠️ Image dimensions too small: {image.shape}")
            return False
        
        return True

# Utility functions for standalone use
def create_classifier(model_path: str, config_path: str = "config.json") -> UnsopleImageClassifier:
    """
    Factory function to create UnsopleImageClassifier instance
    
    Args:
        model_path (str): Path to ONNX model
        config_path (str): Path to config file
        
    Returns:
        UnsopleImageClassifier: Classifier instance
    """
    return UnsopleImageClassifier(model_path, config_path)

def classify_image(image_path: str, model_path: str, config_path: str = "config.json") -> Dict:
    """
    Convenience function for single image classification
    
    Args:
        image_path (str): Path to image file
        model_path (str): Path to ONNX model
        config_path (str): Path to config file
        
    Returns:
        Dict: Classification results
    """
    # Create classifier
    classifier = create_classifier(model_path, config_path)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Perform classification
    prediction, confidence, probabilities = classifier.predict(image)
    
    # Prepare results
    result = {
        'success': prediction is not None,
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': probabilities,
        'bin_recommendation': classifier.get_bin_recommendation(prediction) if prediction else 'unknown',
        'timestamp': time.time()
    }
    
    return result

if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Unsople Image Classifier Test')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    
    args = parser.parse_args()
    
    try:
        result = classify_image(args.image, args.model, args.config)
        
        print("\n" + "="*50)
        print("🧠 UNSOPLE IMAGE CLASSIFICATION RESULT")
        print("="*50)
        
        if result['success']:
            print(f"✅ Waste Type: {result['prediction'].upper()}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print(f"🗑️  Bin Color: {result['bin_recommendation'].upper()}")
            print(f"📈 Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"   - {class_name}: {prob:.3f}")
        else:
            print("❌ No confident prediction")
            print("📈 Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"   - {class_name}: {prob:.3f}")
                
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error: {e}")