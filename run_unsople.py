#!/usr/bin/env python3
"""
🚀 Unsople - AI-Powered Smart Sorting System
📍 Main Entry Point & Application Launcher
🎯 Unified interface for all Unsople functionalities
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from inference import UnsopleInference, main as inference_main
from simulation_test import UnsopleSimulation, main as simulation_main
from utils import UnsopleUtils, setup_logging, get_system_status
from image_classifier import classify_image as single_classify

class UnsopleApplication:
    """
    Main application class for Unsople system
    Provides unified interface for all functionalities
    """
    
    def __init__(self, config_path="config.json"):
        """
        Initialize Unsople application
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = get_config(config_path)
        self.utils = UnsopleUtils(config_path)
        self.logger = setup_logging()
        
        self.inference_system = None
        self.simulation_system = None
        
        self.logger.info("🚀 Unsople Application initialized")

    def run_realtime_inference(self, camera_id=None, save_video=False):
        """
        Run real-time waste classification with camera
        
        Args:
            camera_id (int, optional): Camera device ID
            save_video (bool): Whether to save output video
        """
        try:
            self.logger.info("🎥 Starting real-time inference...")
            
            if self.inference_system is None:
                self.inference_system = UnsopleInference(self.config_path)
            
            # Use configured camera ID if not specified
            if camera_id is None:
                camera_id = self.config['camera']['id']
            
            self.inference_system.run_realtime_inference(
                camera_id=camera_id,
                save_output=save_video
            )
            
        except KeyboardInterrupt:
            self.logger.info("⏹️ Real-time inference stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Real-time inference failed: {e}")
            raise

    def classify_single_image(self, image_path, display=True):
        """
        Classify a single image file
        
        Args:
            image_path (str): Path to image file
            display (bool): Whether to display result
            
        Returns:
            dict: Classification results
        """
        try:
            self.logger.info(f"📷 Classifying single image: {image_path}")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Use the standalone function for simplicity
            result = single_classify(
                image_path=image_path,
                model_path=self.config['model']['path'],
                config_path=self.config_path
            )
            
            if display:
                self._print_classification_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Single image classification failed: {e}")
            raise

    def run_batch_test(self, test_directory, output_dir=None):
        """
        Run batch testing on directory of images
        
        Args:
            test_directory (str): Directory containing test images
            output_dir (str, optional): Output directory for reports
        """
        try:
            self.logger.info(f"📁 Running batch test on: {test_directory}")
            
            if self.simulation_system is None:
                self.simulation_system = UnsopleSimulation(self.config_path)
            
            results = self.simulation_system.run_batch_test(test_directory)
            
            if output_dir:
                report_path = self.simulation_system.generate_test_report(output_dir)
                self.logger.info(f"📊 Test report generated: {report_path}")
            
            self._print_batch_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Batch test failed: {e}")
            raise

    def run_performance_benchmark(self, iterations=100, output_dir=None):
        """
        Run performance benchmark tests
        
        Args:
            iterations (int): Number of test iterations
            output_dir (str, optional): Output directory for reports
        """
        try:
            self.logger.info(f"⚡ Running performance benchmark ({iterations} iterations)")
            
            if self.simulation_system is None:
                self.simulation_system = UnsopleSimulation(self.config_path)
            
            metrics = self.simulation_system.run_performance_benchmark(iterations)
            
            if output_dir:
                report_path = self.simulation_system.generate_test_report(output_dir)
                self.logger.info(f"📊 Performance report generated: {report_path}")
            
            self._print_performance_summary(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Performance benchmark failed: {e}")
            raise

    def run_accuracy_test(self, dataset_path, labels_file=None, output_dir=None):
        """
        Run accuracy test on labeled dataset
        
        Args:
            dataset_path (str): Path to test dataset
            labels_file (str, optional): Path to labels file
            output_dir (str, optional): Output directory for reports
        """
        try:
            self.logger.info(f"🎯 Running accuracy test on: {dataset_path}")
            
            if self.simulation_system is None:
                self.simulation_system = UnsopleSimulation(self.config_path)
            
            accuracy_results = self.simulation_system.run_accuracy_test(
                dataset_path, labels_file
            )
            
            if output_dir:
                report_path = self.simulation_system.generate_test_report(output_dir)
                self.logger.info(f"📊 Accuracy report generated: {report_path}")
            
            self._print_accuracy_summary(accuracy_results)
            
            return accuracy_results
            
        except Exception as e:
            self.logger.error(f"❌ Accuracy test failed: {e}")
            raise

    def show_system_status(self):
        """Display comprehensive system status"""
        try:
            status = get_system_status()
            config = self.config
            
            print("\n" + "="*60)
            print("🖥️  UNSOPLE SYSTEM STATUS")
            print("="*60)
            
            # System Information
            system_info = status['system']
            print("📋 SYSTEM INFORMATION:")
            print(f"   Platform: {system_info['platform']}")
            print(f"   Processor: {system_info['processor']}")
            print(f"   Memory: {system_info['memory_available']:.1f} GB available")
            print(f"   Python: {system_info['python_version']}")
            print(f"   OpenCV: {system_info['opencv_version']}")
            print(f"   GPU Available: {'✅ Yes' if system_info['gpu_available'] else '❌ No'}")
            
            # Memory Usage
            memory_info = status['memory']
            print(f"📊 MEMORY USAGE:")
            print(f"   Process RSS: {memory_info.get('process_rss_mb', 0):.1f} MB")
            print(f"   System Available: {memory_info.get('system_available_gb', 0):.1f} GB")
            
            # Configuration Summary
            print("⚙️  CONFIGURATION SUMMARY:")
            print(f"   Model: {Path(config['model']['path']).name}")
            print(f"   Confidence Threshold: {config['model']['confidence_threshold']}")
            print(f"   Camera: {config['camera']['width']}x{config['camera']['height']} @ {config['camera']['fps']}fps")
            print(f"   Inference Threads: {config['performance']['inference_threads']}")
            
            # Model Information
            model_path = config['model']['path']
            if os.path.exists(model_path):
                model_size = os.path.getsize(model_path) / (1024 * 1024)
                print(f"📦 MODEL INFORMATION:")
                print(f"   Model Size: {model_size:.2f} MB")
                print(f"   Quantized: {'✅ Yes' if config['model']['prefer_quantized'] else '❌ No'}")
            
            print("="*60)
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ System status check failed: {e}")
            raise

    def cleanup_old_results(self, days_to_keep=7):
        """
        Clean up old result directories
        
        Args:
            days_to_keep (int): Number of days to keep results
        """
        try:
            self.logger.info(f"🧹 Cleaning up results older than {days_to_keep} days")
            self.utils.cleanup_old_results(days_to_keep)
            self.logger.info("✅ Cleanup completed")
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")

    def create_backup(self, backup_dir="backups"):
        """
        Create system backup
        
        Args:
            backup_dir (str): Backup directory
        """
        try:
            self.logger.info("💾 Creating system backup...")
            backup_path = self.utils.create_backup(backup_dir)
            if backup_path:
                self.logger.info(f"✅ Backup created: {backup_path}")
            else:
                self.logger.error("❌ Backup creation failed")
        except Exception as e:
            self.logger.error(f"❌ Backup failed: {e}")

    def _print_classification_result(self, result):
        """Print single classification result in formatted way"""
        print("\n" + "="*50)
        print("🧠 UNSOPLE CLASSIFICATION RESULT")
        print("="*50)
        
        if result['success']:
            print(f"✅ Waste Type: {result['prediction'].upper()}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print(f"🗑️  Bin Color: {result['bin_recommendation'].upper()}")
            print(f"🌱 CO2 Saved: {result.get('co2_saved_kg', 0):.3f} kg")
            
            if 'probabilities' in result:
                print(f"📈 Probabilities:")
                for class_name, prob in result['probabilities'].items():
                    print(f"   - {class_name}: {prob:.3f}")
        else:
            print("❌ Classification failed")
            if 'error' in result:
                print(f"   Error: {result['error']}")
        
        print("="*50)

    def _print_batch_summary(self, results):
        """Print batch test summary"""
        successful = len([r for r in results if r.get('predicted_class')])
        total = len(results)
        
        print(f"\n📊 BATCH TEST SUMMARY:")
        print(f"   Total Images: {total}")
        print(f"   Successful Predictions: {successful}")
        print(f"   Success Rate: {successful/total:.1%}" if total > 0 else "N/A")

    def _print_performance_summary(self, metrics):
        """Print performance benchmark summary"""
        stats = metrics['inference_stats']
        
        print(f"\n⚡ PERFORMANCE SUMMARY:")
        print(f"   Mean Inference Time: {stats['mean_ms']:.2f} ms")
        print(f"   FPS: {stats['fps']:.2f}")
        print(f"   Std Deviation: {stats['std_ms']:.2f} ms")
        print(f"   95th Percentile: {stats['percentile_95_ms']:.2f} ms")

    def _print_accuracy_summary(self, accuracy_results):
        """Print accuracy test summary"""
        print(f"\n🎯 ACCURACY SUMMARY:")
        print(f"   Overall Accuracy: {accuracy_results['overall_accuracy']:.2%}")
        print(f"   Total Tests: {accuracy_results['total_tests']}")
        print(f"   Correct Predictions: {accuracy_results['correct_predictions']}")
        print(f"   Average Confidence: {accuracy_results['average_confidence']:.3f}")

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("🧹 Cleaning up Unsople application...")
        
        if self.inference_system:
            # Inference system cleanup would go here
            pass
        
        if self.simulation_system:
            self.simulation_system.cleanup()
        
        self.logger.info("✅ Unsople application cleanup completed")

def main():
    """Main command line interface for Unsople"""
    parser = argparse.ArgumentParser(
        description='🚀 Unsople - AI-Powered Smart Sorting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time classification with camera
  python run_unsople.py --realtime
  
  # Classify single image
  python run_unsople.py --image test.jpg
  
  # Batch test on directory
  python run_unsople.py --batch-test test_images/ --output reports/
  
  # Performance benchmark
  python run_unsople.py --performance --iterations 200
  
  # System status check
  python run_unsople.py --status
  
  # Accuracy test
  python run_unsople.py --accuracy dataset/ --labels labels.json
        """
    )
    
    # Main operation modes
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument('--realtime', action='store_true', 
                               help='Run real-time camera inference')
    operation_group.add_argument('--image', type=str, metavar='PATH',
                               help='Classify single image file')
    operation_group.add_argument('--batch-test', type=str, metavar='DIRECTORY',
                               help='Run batch test on directory of images')
    operation_group.add_argument('--performance', action='store_true',
                               help='Run performance benchmark')
    operation_group.add_argument('--accuracy', type=str, metavar='DATASET_PATH',
                               help='Run accuracy test on labeled dataset')
    operation_group.add_argument('--status', action='store_true',
                               help='Show system status')
    operation_group.add_argument('--cleanup', type=int, metavar='DAYS', nargs='?', const=7,
                               help='Cleanup old results (default: 7 days)')
    operation_group.add_argument('--backup', action='store_true',
                               help='Create system backup')
    
    # Optional arguments
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--save-video', action='store_true',
                       help='Save video output in real-time mode')
    parser.add_argument('--output', type=str,
                       help='Output directory for test reports')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Iterations for performance test (default: 100)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config file (default: config.json)')
    parser.add_argument('--labels', type=str,
                       help='Labels file for accuracy test')
    
    args = parser.parse_args()
    
    # Initialize application
    app = None
    try:
        app = UnsopleApplication(args.config)
        
        if args.realtime:
            app.run_realtime_inference(
                camera_id=args.camera,
                save_video=args.save_video
            )
        
        elif args.image:
            app.classify_single_image(args.image, display=True)
        
        elif args.batch_test:
            app.run_batch_test(args.batch_test, args.output)
        
        elif args.performance:
            app.run_performance_benchmark(args.iterations, args.output)
        
        elif args.accuracy:
            app.run_accuracy_test(args.accuracy, args.labels, args.output)
        
        elif args.status:
            app.show_system_status()
        
        elif args.cleanup:
            app.cleanup_old_results(args.cleanup)
        
        elif args.backup:
            app.create_backup()
        
        print("\n✅ Unsople operation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unsople operation failed: {e}")
        sys.exit(1)
    finally:
        if app:
            app.cleanup()

if __name__ == "__main__":
    main()