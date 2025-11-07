# simulation_test.py
"""
🧪 Unsople - AI-Powered Smart Sorting System
📍 Simulation & Testing Module
🎯 Comprehensive testing and simulation for waste classification system
"""

import os
import cv2
import time
import json
import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import sys

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_classifier import UnsopleImageClassifier
from impact_calculator import CO2ImpactCalculator
from utils import UnsopleUtils, setup_logging, get_config

class UnsopleSimulation:
    """
    Comprehensive testing and simulation environment for Unsople system
    Supports batch testing, performance benchmarking, and accuracy validation
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize simulation environment
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = get_config(config_path)
        self.utils = UnsopleUtils(config_path)
        self.logger = setup_logging()
        
        # Initialize components
        self.classifier = None
        self.impact_calculator = None
        
        # Test results storage
        self.test_results = []
        self.performance_metrics = {}
        
        # Simulation state
        self.is_running = False
        self.start_time = None
        
        self.logger.info("✅ Unsople Simulation environment initialized")
    
    def initialize_components(self):
        """Initialize AI components for testing"""
        try:
            self.logger.info("🧠 Initializing AI components...")
            
            # Initialize classifier
            model_path = self.config['model']['path']
            self.classifier = UnsopleImageClassifier(model_path, self.config_path)
            
            # Initialize impact calculator
            self.impact_calculator = CO2ImpactCalculator(self.config_path)
            
            self.logger.info("✅ AI components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            return False
    
    def run_single_test(self, image_path: str, expected_class: Optional[str] = None) -> Dict:
        """
        Run single image test
        
        Args:
            image_path (str): Path to test image
            expected_class (str, optional): Expected class for accuracy calculation
            
        Returns:
            Dict: Test results
        """
        try:
            if self.classifier is None:
                self.initialize_components()
            
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            is_valid, message = self.utils.validate_image(image)
            if not is_valid:
                raise ValueError(f"Invalid image: {message}")
            
            # Perform classification
            start_time = time.time()
            prediction, confidence, probabilities = self.classifier.predict(image)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Calculate impact if prediction is successful
            impact_data = {}
            if prediction:
                impact_data = self.impact_calculator.calculate_impact(prediction)
            
            # Determine test result
            is_correct = False
            if expected_class and prediction:
                is_correct = (prediction.lower() == expected_class.lower())
            
            # Compile results
            result = {
                'image_path': image_path,
                'expected_class': expected_class,
                'predicted_class': prediction,
                'confidence': confidence,
                'inference_time_ms': inference_time,
                'is_correct': is_correct,
                'probabilities': probabilities,
                'impact_data': impact_data,
                'timestamp': datetime.now().isoformat(),
                'image_size': f"{image.shape[1]}x{image.shape[0]}"
            }
            
            self.logger.debug(f"🧪 Single test: {prediction} (confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Single test failed for {image_path}: {e}")
            return {
                'image_path': image_path,
                'expected_class': expected_class,
                'predicted_class': None,
                'confidence': 0.0,
                'inference_time_ms': 0.0,
                'is_correct': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_batch_test(self, test_directory: str, 
                      expected_classes: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Run batch test on directory of images
        
        Args:
            test_directory (str): Directory containing test images
            expected_classes (Dict, optional): Mapping of filename to expected class
            
        Returns:
            List[Dict]: Batch test results
        """
        self.logger.info(f"📁 Running batch test on directory: {test_directory}")
        
        if self.classifier is None:
            self.initialize_components()
        
        test_dir = Path(test_directory)
        if not test_dir.exists():
            raise ValueError(f"Test directory not found: {test_directory}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(test_dir.glob(f"*{ext}"))
            image_files.extend(test_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {test_directory}")
        
        self.logger.info(f"📸 Found {len(image_files)} test images")
        
        # Run tests
        results = []
        for image_path in tqdm(image_files, desc="Running batch tests"):
            expected_class = None
            if expected_classes:
                expected_class = expected_classes.get(image_path.name)
            
            result = self.run_single_test(str(image_path), expected_class)
            results.append(result)
        
        self.test_results.extend(results)
        self.logger.info(f"✅ Batch test completed: {len(results)} images processed")
        
        return results
    
    def run_performance_benchmark(self, num_iterations: int = 100, 
                                batch_size: int = 1) -> Dict:
        """
        Run comprehensive performance benchmark
        
        Args:
            num_iterations (int): Number of test iterations
            batch_size (int): Batch size for testing
            
        Returns:
            Dict: Performance metrics
        """
        self.logger.info(f"⚡ Running performance benchmark ({num_iterations} iterations)")
        
        if self.classifier is None:
            self.initialize_components()
        
        # Generate test data
        input_size = self.config['processing']['input_size']
        test_images = [
            np.random.randint(0, 255, (input_size[0], input_size[1], 3), dtype=np.uint8)
            for _ in range(num_iterations)
        ]
        
        # Warmup
        self.logger.debug("🔥 Warming up...")
        for i in range(min(10, num_iterations)):
            _ = self.classifier.predict(test_images[i])
        
        # Benchmark inference speed
        inference_times = []
        memory_usage = []
        
        self.logger.debug("⏱️ Benchmarking inference speed...")
        for i in tqdm(range(num_iterations), desc="Performance testing"):
            # Measure memory before
            mem_before = self.utils.get_memory_usage()
            
            # Time inference
            start_time = time.time()
            prediction, confidence, _ = self.classifier.predict(test_images[i])
            inference_time = (time.time() - start_time) * 1000
            
            # Measure memory after
            mem_after = self.utils.get_memory_usage()
            
            inference_times.append(inference_time)
            memory_usage.append(mem_after.get('process_rss_mb', 0))
        
        # Calculate statistics
        inference_stats = {
            'mean_ms': np.mean(inference_times),
            'median_ms': np.median(inference_times),
            'std_ms': np.std(inference_times),
            'min_ms': np.min(inference_times),
            'max_ms': np.max(inference_times),
            'fps': 1000 / np.mean(inference_times),
            'percentile_95_ms': np.percentile(inference_times, 95),
            'percentile_99_ms': np.percentile(inference_times, 99)
        }
        
        memory_stats = {
            'mean_mb': np.mean(memory_usage),
            'max_mb': np.max(memory_usage),
            'min_mb': np.min(memory_usage)
        }
        
        # System information
        system_info = self.utils.get_system_info()
        
        performance_metrics = {
            'inference_times': inference_times,
            'inference_stats': inference_stats,
            'memory_usage': memory_usage,
            'memory_stats': memory_stats,
            'system_info': system_info.__dict__,
            'test_parameters': {
                'num_iterations': num_iterations,
                'batch_size': batch_size,
                'input_size': input_size
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self.performance_metrics = performance_metrics
        self.logger.info(f"✅ Performance benchmark completed: "
                        f"{inference_stats['mean_ms']:.2f} ms average inference time")
        
        return performance_metrics
    
    def run_accuracy_test(self, test_dataset_path: str, 
                         labels_file: Optional[str] = None) -> Dict:
        """
        Run accuracy test on labeled dataset
        
        Args:
            test_dataset_path (str): Path to test dataset
            labels_file (str, optional): Path to labels file
            
        Returns:
            Dict: Accuracy test results
        """
        self.logger.info(f"🎯 Running accuracy test on: {test_dataset_path}")
        
        if self.classifier is None:
            self.initialize_components()
        
        dataset_path = Path(test_dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path not found: {test_dataset_path}")
        
        # Load labels if provided
        labels = {}
        if labels_file and Path(labels_file).exists():
            with open(labels_file, 'r') as f:
                labels_data = json.load(f)
                labels = labels_data.get('labels', {})
        
        # If no labels file, assume directory structure represents classes
        if not labels and dataset_path.is_dir():
            labels = {}
            for class_dir in dataset_path.iterdir():
                if class_dir.is_dir():
                    for image_file in class_dir.glob('*.*'):
                        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            labels[image_file.name] = class_dir.name
        
        # Run tests
        results = []
        total_tests = len(labels)
        
        self.logger.info(f"📊 Testing {total_tests} labeled images")
        
        for filename, expected_class in tqdm(labels.items(), desc="Accuracy testing"):
            image_path = None
            
            # Find image file
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_path = dataset_path / filename
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if not image_path:
                # Try to find in subdirectories
                for subdir in dataset_path.iterdir():
                    if subdir.is_dir():
                        potential_path = subdir / filename
                        if potential_path.exists():
                            image_path = potential_path
                            break
            
            if image_path and image_path.exists():
                result = self.run_single_test(str(image_path), expected_class)
                results.append(result)
            else:
                self.logger.warning(f"⚠️ Could not find image: {filename}")
        
        # Calculate accuracy metrics
        correct_predictions = sum(1 for r in results if r.get('is_correct', False))
        total_predictions = len([r for r in results if r.get('predicted_class')])
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Per-class accuracy
        class_metrics = self._calculate_class_metrics(results)
        
        accuracy_results = {
            'total_tests': len(results),
            'successful_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'overall_accuracy': accuracy,
            'class_metrics': class_metrics,
            'average_confidence': np.mean([r.get('confidence', 0) for r in results if r.get('confidence')]),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"✅ Accuracy test completed: {accuracy:.2%} overall accuracy")
        
        return accuracy_results
    
    def _calculate_class_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate per-class accuracy metrics
        
        Args:
            results (List[Dict]): Test results
            
        Returns:
            Dict: Class-wise metrics
        """
        class_data = {}
        
        for result in results:
            expected = result.get('expected_class')
            predicted = result.get('predicted_class')
            is_correct = result.get('is_correct', False)
            
            if expected and predicted:
                if expected not in class_data:
                    class_data[expected] = {
                        'total': 0,
                        'correct': 0,
                        'predictions': {}
                    }
                
                class_data[expected]['total'] += 1
                if is_correct:
                    class_data[expected]['correct'] += 1
                
                # Track what this class was confused with
                if predicted not in class_data[expected]['predictions']:
                    class_data[expected]['predictions'][predicted] = 0
                class_data[expected]['predictions'][predicted] += 1
        
        # Calculate metrics for each class
        class_metrics = {}
        for class_name, data in class_data.items():
            accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
            
            class_metrics[class_name] = {
                'total_samples': data['total'],
                'correct_predictions': data['correct'],
                'accuracy': accuracy,
                'common_confusions': dict(sorted(
                    data['predictions'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3])  # Top 3 confusions
            }
        
        return class_metrics
    
    def run_stress_test(self, duration_seconds: int = 300) -> Dict:
        """
        Run stress test to check system stability
        
        Args:
            duration_seconds (int): Test duration in seconds
            
        Returns:
            Dict: Stress test results
        """
        self.logger.info(f"💥 Running stress test for {duration_seconds} seconds")
        
        if self.classifier is None:
            self.initialize_components()
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        test_results = {
            'total_iterations': 0,
            'successful_iterations': 0,
            'failed_iterations': 0,
            'inference_times': [],
            'memory_usage': [],
            'errors': []
        }
        
        iteration = 0
        with tqdm(total=duration_seconds, desc="Stress testing") as pbar:
            while time.time() < end_time:
                iteration += 1
                elapsed = time.time() - start_time
                pbar.update(int(elapsed) - pbar.n)
                
                try:
                    # Generate random test image
                    input_size = self.config['processing']['input_size']
                    test_image = np.random.randint(
                        0, 255, (input_size[0], input_size[1], 3), dtype=np.uint8
                    )
                    
                    # Measure memory before
                    mem_before = self.utils.get_memory_usage()
                    
                    # Run inference
                    inference_start = time.time()
                    prediction, confidence, _ = self.classifier.predict(test_image)
                    inference_time = (time.time() - inference_start) * 1000
                    
                    # Measure memory after
                    mem_after = self.utils.get_memory_usage()
                    
                    test_results['total_iterations'] += 1
                    test_results['successful_iterations'] += 1
                    test_results['inference_times'].append(inference_time)
                    test_results['memory_usage'].append(
                        mem_after.get('process_rss_mb', 0)
                    )
                    
                    # Small delay to prevent overheating
                    time.sleep(0.01)
                    
                except Exception as e:
                    test_results['failed_iterations'] += 1
                    test_results['errors'].append({
                        'iteration': iteration,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    self.logger.warning(f"⚠️ Stress test iteration {iteration} failed: {e}")
        
        # Calculate statistics
        if test_results['inference_times']:
            test_results['inference_stats'] = {
                'mean_ms': np.mean(test_results['inference_times']),
                'std_ms': np.std(test_results['inference_times']),
                'min_ms': np.min(test_results['inference_times']),
                'max_ms': np.max(test_results['inference_times'])
            }
        
        if test_results['memory_usage']:
            test_results['memory_stats'] = {
                'mean_mb': np.mean(test_results['memory_usage']),
                'max_mb': np.max(test_results['memory_usage']),
                'min_mb': np.min(test_results['memory_usage'])
            }
        
        test_results['success_rate'] = (
            test_results['successful_iterations'] / test_results['total_iterations']
            if test_results['total_iterations'] > 0 else 0
        )
        
        self.logger.info(f"✅ Stress test completed: "
                        f"{test_results['success_rate']:.2%} success rate")
        
        return test_results
    
    def generate_test_report(self, output_dir: Optional[str] = None) -> str:
        """
        Generate comprehensive test report
        
        Args:
            output_dir (str, optional): Output directory for report
            
        Returns:
            str: Path to generated report
        """
        if output_dir is None:
            output_dir = self.utils.create_results_directory()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📊 Generating test report in: {output_dir}")
        
        # Generate various report components
        report_data = {
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'system_info': self.utils.get_system_info().__dict__,
            'config': self.config,
            'generation_time': datetime.now().isoformat()
        }
        
        # Save JSON report
        json_report_path = output_dir / "unsople_test_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # Generate visualizations if we have data
        if self.test_results:
            self._generate_accuracy_plots(output_dir)
        
        if self.performance_metrics:
            self._generate_performance_plots(output_dir)
        
        # Generate summary CSV
        self._generate_summary_csv(output_dir)
        
        self.logger.info(f"✅ Test report generated: {json_report_path}")
        return str(json_report_path)
    
    def _generate_accuracy_plots(self, output_dir: Path):
        """Generate accuracy visualization plots"""
        try:
            if not self.test_results:
                return
            
            # Filter successful predictions
            successful_results = [r for r in self.test_results if r.get('predicted_class')]
            
            if not successful_results:
                return
            
            # Accuracy by class
            class_accuracy = {}
            for result in successful_results:
                expected = result.get('expected_class')
                if expected and result.get('is_correct') is not None:
                    if expected not in class_accuracy:
                        class_accuracy[expected] = {'total': 0, 'correct': 0}
                    
                    class_accuracy[expected]['total'] += 1
                    if result['is_correct']:
                        class_accuracy[expected]['correct'] += 1
            
            # Calculate accuracy percentages
            classes = list(class_accuracy.keys())
            accuracies = [
                class_accuracy[cls]['correct'] / class_accuracy[cls]['total'] 
                for cls in classes
            ]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.7)
            plt.title('Accuracy by Waste Class', fontsize=14, fontweight='bold')
            plt.xlabel('Waste Class', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, accuracy in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{accuracy:.2%}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'accuracy_by_class.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not generate accuracy plots: {e}")
    
    def _generate_performance_plots(self, output_dir: Path):
        """Generate performance visualization plots"""
        try:
            if not self.performance_metrics.get('inference_times'):
                return
            
            inference_times = self.performance_metrics['inference_times']
            
            # Inference time distribution
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.hist(inference_times, bins=50, alpha=0.7, color='lightcoral')
            plt.title('Inference Time Distribution', fontweight='bold')
            plt.xlabel('Inference Time (ms)')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            plt.plot(inference_times, alpha=0.7)
            plt.title('Inference Time Over Time', fontweight='bold')
            plt.xlabel('Iteration')
            plt.ylabel('Inference Time (ms)')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not generate performance plots: {e}")
    
    def _generate_summary_csv(self, output_dir: Path):
        """Generate summary CSV report"""
        try:
            if not self.test_results:
                return
            
            csv_path = output_dir / "test_summary.csv"
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Image Path', 'Expected Class', 'Predicted Class', 
                    'Confidence', 'Inference Time (ms)', 'Is Correct',
                    'Timestamp'
                ])
                
                for result in self.test_results:
                    writer.writerow([
                        result.get('image_path', ''),
                        result.get('expected_class', ''),
                        result.get('predicted_class', ''),
                        f"{result.get('confidence', 0):.4f}",
                        f"{result.get('inference_time_ms', 0):.2f}",
                        result.get('is_correct', False),
                        result.get('timestamp', '')
                    ])
            
            self.logger.info(f"💾 Summary CSV saved: {csv_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not generate summary CSV: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        self.logger.info("🧹 Simulation cleanup completed")

def main():
    """Main function for command line testing"""
    parser = argparse.ArgumentParser(description='Unsople Simulation & Testing')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'batch', 'performance', 'accuracy', 'stress'],
                       help='Test mode')
    parser.add_argument('--image', type=str, help='Image path for single test')
    parser.add_argument('--directory', type=str, help='Directory for batch test')
    parser.add_argument('--labels', type=str, help='Labels file for accuracy test')
    parser.add_argument('--iterations', type=int, default=100, 
                       help='Iterations for performance test')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration in seconds for stress test')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config file')
    parser.add_argument('--output', type=str, help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Initialize simulation
    simulation = UnsopleSimulation(args.config)
    
    try:
        if args.mode == 'single':
            if not args.image:
                print("❌ Please provide --image for single test mode")
                return
            
            result = simulation.run_single_test(args.image)
            print(f"\n🧪 SINGLE TEST RESULT:")
            print(f"   Image: {args.image}")
            print(f"   Prediction: {result.get('predicted_class', 'None')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Inference Time: {result.get('inference_time_ms', 0):.2f} ms")
            
            if result.get('is_correct') is not None:
                print(f"   Correct: {result['is_correct']}")
        
        elif args.mode == 'batch':
            if not args.directory:
                print("❌ Please provide --directory for batch test mode")
                return
            
            results = simulation.run_batch_test(args.directory)
            print(f"\n📊 BATCH TEST COMPLETED:")
            print(f"   Total Images: {len(results)}")
            
            successful = len([r for r in results if r.get('predicted_class')])
            print(f"   Successful Predictions: {successful}")
            
            if args.output:
                report_path = simulation.generate_test_report(args.output)
                print(f"   Report: {report_path}")
        
        elif args.mode == 'performance':
            metrics = simulation.run_performance_benchmark(args.iterations)
            stats = metrics['inference_stats']
            
            print(f"\n⚡ PERFORMANCE BENCHMARK:")
            print(f"   Mean Inference Time: {stats['mean_ms']:.2f} ms")
            print(f"   FPS: {stats['fps']:.2f}")
            print(f"   Std Dev: {stats['std_ms']:.2f} ms")
            print(f"   95th Percentile: {stats['percentile_95_ms']:.2f} ms")
            
            if args.output:
                report_path = simulation.generate_test_report(args.output)
                print(f"   Report: {report_path}")
        
        elif args.mode == 'accuracy':
            if not args.directory:
                print("❌ Please provide --directory for accuracy test mode")
                return
            
            accuracy_results = simulation.run_accuracy_test(args.directory, args.labels)
            
            print(f"\n🎯 ACCURACY TEST RESULTS:")
            print(f"   Overall Accuracy: {accuracy_results['overall_accuracy']:.2%}")
            print(f"   Total Tests: {accuracy_results['total_tests']}")
            print(f"   Correct Predictions: {accuracy_results['correct_predictions']}")
            print(f"   Average Confidence: {accuracy_results['average_confidence']:.3f}")
            
            if args.output:
                report_path = simulation.generate_test_report(args.output)
                print(f"   Report: {report_path}")
        
        elif args.mode == 'stress':
            stress_results = simulation.run_stress_test(args.duration)
            
            print(f"\n💥 STRESS TEST RESULTS:")
            print(f"   Total Iterations: {stress_results['total_iterations']}")
            print(f"   Successful: {stress_results['successful_iterations']}")
            print(f"   Failed: {stress_results['failed_iterations']}")
            print(f"   Success Rate: {stress_results['success_rate']:.2%}")
            
            if stress_results.get('inference_stats'):
                stats = stress_results['inference_stats']
                print(f"   Mean Inference Time: {stats['mean_ms']:.2f} ms")
        
        print(f"\n✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        sys.exit(1)
    
    finally:
        simulation.cleanup()

if __name__ == "__main__":
    main()