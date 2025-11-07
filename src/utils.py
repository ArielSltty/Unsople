# utils.py
"""
🔧 Unsople - AI-Powered Smart Sorting System
📍 Utility Functions & Helper Modules
🎯 Common utilities for configuration, logging, file handling, and system operations
"""

import os
import json
import logging
import csv
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import platform
import psutil
import GPUtil
from dataclasses import dataclass, asdict
from enum import Enum
import argparse

class LogLevel(Enum):
    """Logging level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SystemStatus(Enum):
    """System status enumeration"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class SystemInfo:
    """System information data class"""
    platform: str
    processor: str
    memory_total: float
    memory_available: float
    disk_usage: float
    python_version: str
    opencv_version: str
    gpu_available: bool
    gpu_info: Optional[Dict]

@dataclass
class AppConfig:
    """Application configuration data class"""
    # Model settings
    model_path: str
    model_config_path: str
    confidence_threshold: float
    
    # Camera settings
    camera_id: int
    camera_width: int
    camera_height: int
    camera_fps: int
    
    # Processing settings
    input_size: Tuple[int, int]
    mean_values: List[float]
    std_values: List[float]
    
    # Output settings
    results_directory: str
    save_detected_images: bool
    save_video_output: bool
    
    # Performance settings
    inference_threads: int
    enable_optimizations: bool
    
    # Impact calculation settings
    co2_calculation_enabled: bool
    auto_save_reports: bool

class UnsopleUtils:
    """
    Utility class for Unsople system operations
    Handles configuration, logging, file operations, and system monitoring
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "model": {
            "path": "models/unsople_waste_classifier.onnx",
            "config_path": "models/unsople_model_config.json",
            "confidence_threshold": 0.6
        },
        "camera": {
            "id": 0,
            "width": 640,
            "height": 480,
            "fps": 30
        },
        "processing": {
            "input_size": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "auto_orient": True,
            "maintain_aspect_ratio": True
        },
        "output": {
            "results_directory": "results",
            "save_detected_images": True,
            "save_video_output": False,
            "image_quality": 95
        },
        "performance": {
            "inference_threads": 4,
            "enable_optimizations": True,
            "memory_limit_mb": 1024
        },
        "impact": {
            "co2_calculation_enabled": True,
            "auto_save_reports": True,
            "daily_summary": True
        },
        "logging": {
            "level": "INFO",
            "file_enabled": True,
            "console_enabled": True,
            "max_file_size_mb": 10
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize utilities
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.logger = None
        self._setup_logging()

    def _setup_logging(self):
        """Setup comprehensive logging system"""
        logger = logging.getLogger('Unsople')
        logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.config['logging']['console_enabled']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.config['logging']['file_enabled']:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"unsople_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        self.logger = logger
        self.logger.info("✅ Logging system initialized")

    def load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration from JSON file with fallbacks
        
        Args:
            config_path (str, optional): Path to config file
            
        Returns:
            Dict: Configuration dictionary
        """
        if config_path is None:
            config_path = self.config_path or "config.json"
        
        try:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Deep merge with defaults
                config = self._deep_merge(self.DEFAULT_CONFIG, user_config)
                self.logger.info(f"📁 Configuration loaded from {config_path}")
            else:
                config = self.DEFAULT_CONFIG.copy()
                self.logger.warning(f"⚠️ Config file not found, using defaults")
                
                # Save default config
                self.save_config(config, config_path)
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load config: {e}")
            return self.DEFAULT_CONFIG.copy()

    def save_config(self, config: Dict, config_path: Optional[str] = None) -> bool:
        """
        Save configuration to JSON file
        
        Args:
            config (Dict): Configuration dictionary
            config_path (str, optional): Path to save config
            
        Returns:
            bool: Success status
        """
        try:
            if config_path is None:
                config_path = self.config_path or "config.json"
            
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save config: {e}")
            return False

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Deep merge two dictionaries
        
        Args:
            base (Dict): Base dictionary
            update (Dict): Update dictionary
            
        Returns:
            Dict: Merged dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def get_system_info(self) -> SystemInfo:
        """
        Get comprehensive system information
        
        Returns:
            SystemInfo: System information object
        """
        try:
            # Platform information
            system_platform = platform.system()
            processor = platform.processor()
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024 ** 3)
            memory_available_gb = memory.available / (1024 ** 3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # GPU information
            gpu_available = False
            gpu_info = None
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_available = True
                    gpu = gpus[0]
                    gpu_info = {
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed,
                        'load': gpu.load * 100
                    }
            except Exception:
                pass  # GPU info not available
            
            return SystemInfo(
                platform=system_platform,
                processor=processor,
                memory_total=memory_total_gb,
                memory_available=memory_available_gb,
                disk_usage=disk_usage_percent,
                python_version=sys.version,
                opencv_version=cv2.__version__,
                gpu_available=gpu_available,
                gpu_info=gpu_info
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get system info: {e}")
            return SystemInfo(
                platform="Unknown",
                processor="Unknown",
                memory_total=0,
                memory_available=0,
                disk_usage=0,
                python_version=sys.version,
                opencv_version=cv2.__version__,
                gpu_available=False,
                gpu_info=None
            )

    def create_results_directory(self, base_dir: Optional[str] = None) -> Path:
        """
        Create results directory with timestamp
        
        Args:
            base_dir (str, optional): Base directory path
            
        Returns:
            Path: Path to results directory
        """
        if base_dir is None:
            base_dir = self.config['output']['results_directory']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(base_dir) / f"unsople_results_{timestamp}"
        
        # Create directory structure
        directories = [
            results_dir,
            results_dir / "detected_images",
            results_dir / "logs",
            results_dir / "reports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📁 Results directory created: {results_dir}")
        return results_dir

    def save_detection_image(self, image: np.ndarray, prediction: str, 
                           confidence: float, output_dir: Path) -> Optional[Path]:
        """
        Save detection image with metadata
        
        Args:
            image (np.ndarray): Image to save
            prediction (str): Prediction class
            confidence (float): Confidence score
            output_dir (Path): Output directory
            
        Returns:
            Path: Path to saved image or None if failed
        """
        try:
            if not self.config['output']['save_detected_images']:
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}_{prediction}_{confidence:.2f}.jpg"
            output_path = output_dir / filename
            
            # Add annotation to image
            annotated_image = self._annotate_image(image, prediction, confidence)
            
            # Save with specified quality
            quality = self.config['output']['image_quality']
            cv2.imwrite(str(output_path), annotated_image, 
                       [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            self.logger.debug(f"💾 Detection image saved: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save detection image: {e}")
            return None

    def _annotate_image(self, image: np.ndarray, prediction: str, 
                       confidence: float) -> np.ndarray:
        """
        Annotate image with prediction information
        
        Args:
            image (np.ndarray): Input image
            prediction (str): Prediction class
            confidence (float): Confidence score
            
        Returns:
            np.ndarray: Annotated image
        """
        annotated = image.copy()
        
        # Convert to PIL for easier text rendering
        pil_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        try:
            # Try to use a nice font, fallback to default
            font_size = max(20, min(image.shape[1] // 20, 40))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Text content
        text = f"{prediction.upper()} ({confidence:.1%})"
        
        # Get text bounds
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(text, font=font)
        
        # Calculate position (top center)
        x = (image.shape[1] - text_width) // 2
        y = 10
        
        # Draw background
        padding = 10
        draw.rectangle([
            x - padding, y - padding,
            x + text_width + padding, y + text_height + padding
        ], fill=(0, 0, 0, 128))
        
        # Draw text
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def setup_camera(self, camera_id: Optional[int] = None) -> cv2.VideoCapture:
        """
        Setup camera with configuration
        
        Args:
            camera_id (int, optional): Camera device ID
            
        Returns:
            cv2.VideoCapture: Camera object
        """
        if camera_id is None:
            camera_id = self.config['camera']['id']
        
        try:
            cap = cv2.VideoCapture(camera_id)
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
            cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            
            # Try to set auto exposure and white balance
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Auto white balance
            
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {camera_id}")
            
            # Test camera
            ret, test_frame = cap.read()
            if not ret:
                raise RuntimeError("Camera test frame failed")
            
            self.logger.info(f"📷 Camera {camera_id} initialized: "
                           f"{test_frame.shape[1]}x{test_frame.shape[0]}")
            
            return cap
            
        except Exception as e:
            self.logger.error(f"❌ Camera setup failed: {e}")
            raise

    def validate_image(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image for processing
        
        Args:
            image (np.ndarray): Image to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if image is None:
            return False, "Image is None"
        
        if image.size == 0:
            return False, "Image is empty"
        
        if len(image.shape) not in [2, 3]:
            return False, f"Invalid image shape: {image.shape}"
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False, f"Invalid number of channels: {image.shape[2]}"
        
        min_dimension = 32
        if image.shape[0] < min_dimension or image.shape[1] < min_dimension:
            return False, f"Image too small: {image.shape}"
        
        return True, "Valid"

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics
        
        Returns:
            Dict: Memory usage information
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            system_memory = psutil.virtual_memory()
            
            return {
                'process_rss_mb': memory_info.rss / (1024 ** 2),
                'process_vms_mb': memory_info.vms / (1024 ** 2),
                'system_available_gb': system_memory.available / (1024 ** 3),
                'system_used_percent': system_memory.percent
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get memory usage: {e}")
            return {}

    def performance_monitor(self, func):
        """
        Decorator to monitor function performance
        
        Args:
            func: Function to monitor
            
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                end_memory = self.get_memory_usage()
                
                # Log performance
                self.logger.debug(
                    f"⏱️ {func.__name__} executed in {execution_time:.3f}s, "
                    f"Memory: {end_memory.get('process_rss_mb', 0):.1f}MB"
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(
                    f"❌ {func.__name__} failed after {execution_time:.3f}s: {e}"
                )
                raise
        
        return wrapper

    def cleanup_old_results(self, days_to_keep: int = 7):
        """
        Clean up old result directories
        
        Args:
            days_to_keep (int): Number of days to keep results
        """
        try:
            base_dir = Path(self.config['output']['results_directory'])
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            
            deleted_count = 0
            for result_dir in base_dir.glob("unsople_results_*"):
                if result_dir.is_dir():
                    # Extract timestamp from directory name
                    dir_time_str = result_dir.name.replace("unsople_results_", "")
                    try:
                        dir_time = datetime.strptime(dir_time_str, "%Y%m%d_%H%M%S")
                        if dir_time < cutoff_time:
                            import shutil
                            shutil.rmtree(result_dir)
                            deleted_count += 1
                            self.logger.info(f"🗑️ Cleaned up old results: {result_dir}")
                    except ValueError:
                        continue  # Skip directories with invalid names
            
            self.logger.info(f"🧹 Cleanup completed: {deleted_count} directories removed")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")

    def create_backup(self, backup_dir: Optional[str] = None) -> Optional[Path]:
        """
        Create backup of important files
        
        Args:
            backup_dir (str, optional): Backup directory
            
        Returns:
            Path: Path to backup directory
        """
        try:
            if backup_dir is None:
                backup_dir = "backups"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(backup_dir) / f"unsople_backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Files to backup
            backup_files = [
                "config.json",
                "models/unsople_model_config.json",
                "results/impact_report.csv",
                "results/classification_log.csv"
            ]
            
            backed_up_count = 0
            for file_pattern in backup_files:
                for file_path in Path('.').glob(file_pattern):
                    if file_path.exists():
                        relative_path = file_path.relative_to('.')
                        backup_file_path = backup_path / relative_path
                        backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        import shutil
                        shutil.copy2(file_path, backup_file_path)
                        backed_up_count += 1
            
            self.logger.info(f"💾 Backup created: {backup_path} ({backed_up_count} files)")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"❌ Backup failed: {e}")
            return None

# Global utility functions
def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Quick setup for logging
    
    Args:
        level (str): Logging level
        
    Returns:
        logging.Logger: Logger instance
    """
    utils = UnsopleUtils()
    return utils.logger

def get_config(config_path: Optional[str] = None) -> Dict:
    """
    Quick config loader
    
    Args:
        config_path (str, optional): Path to config file
        
    Returns:
        Dict: Configuration dictionary
    """
    utils = UnsopleUtils()
    return utils.load_config(config_path)

def create_results_directory(base_dir: Optional[str] = None) -> Path:
    """
    Quick results directory creator
    
    Args:
        base_dir (str, optional): Base directory
        
    Returns:
        Path: Results directory path
    """
    utils = UnsopleUtils()
    return utils.create_results_directory(base_dir)

def validate_image_file(image_path: str) -> Tuple[bool, str]:
    """
    Validate image file
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False, "Could not read image file"
        
        utils = UnsopleUtils()
        return utils.validate_image(image)
        
    except Exception as e:
        return False, f"Validation error: {e}"

def get_system_status() -> Dict:
    """
    Get comprehensive system status
    
    Returns:
        Dict: System status information
    """
    utils = UnsopleUtils()
    system_info = utils.get_system_info()
    memory_usage = utils.get_memory_usage()
    
    return {
        'system': asdict(system_info),
        'memory': memory_usage,
        'timestamp': datetime.now().isoformat(),
        'status': SystemStatus.READY.value
    }

if __name__ == "__main__":
    # Command line interface for utilities
    parser = argparse.ArgumentParser(description='Unsople Utilities')
    parser.add_argument('--system-info', action='store_true', help='Show system information')
    parser.add_argument('--validate-image', type=str, help='Validate an image file')
    parser.add_argument('--create-backup', action='store_true', help='Create system backup')
    parser.add_argument('--cleanup', type=int, default=7, help='Cleanup old results (days to keep)')
    
    args = parser.parse_args()
    
    utils = UnsopleUtils()
    
    if args.system_info:
        system_info = utils.get_system_info()
        print("\n" + "="*50)
        print("🖥️  UNSOPLE SYSTEM INFORMATION")
        print("="*50)
        for key, value in asdict(system_info).items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print("="*50)
    
    elif args.validate_image:
        is_valid, message = validate_image_file(args.validate_image)
        print(f"\n📷 Image Validation: {args.validate_image}")
        print(f"   Status: {'✅ VALID' if is_valid else '❌ INVALID'}")
        print(f"   Message: {message}")
    
    elif args.create_backup:
        backup_path = utils.create_backup()
        if backup_path:
            print(f"✅ Backup created: {backup_path}")
        else:
            print("❌ Backup failed")
    
    elif args.cleanup:
        utils.cleanup_old_results(args.cleanup)
        print(f"✅ Cleanup completed (keeping {args.cleanup} days of results)")
    
    else:
        parser.print_help()