# inference.py
"""
🚀 Unsople - AI-Powered Smart Sorting System
📍 Real-time Waste Classification Inference Module
🎯 Main script for camera inference and image classification
"""

import cv2
import argparse
import time
import json
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_classifier import UnsopleImageClassifier
from impact_calculator import CO2ImpactCalculator
from utils import setup_logging, get_config, create_results_directory

class UnsopleInference:
    """
    Main inference class for Unsople waste classification system
    Handles real-time camera inference and image classification
    """
    
    def __init__(self, config_path="config.json"):
        """
        Initialize Unsople inference system
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        self.config = get_config(config_path)
        
        # Setup results directory
        self.results_dir = create_results_directory()
        
        # Initialize components
        self.classifier = UnsopleImageClassifier(
            model_path=self.config['model_path'],
            config_path=config_path
        )
        
        self.impact_calculator = CO2ImpactCalculator(config_path)
        
        # Setup logging
        self.logger = setup_logging()
        
        # Inference statistics
        self.stats = {
            'total_predictions': 0,
            'total_co2_saved': 0.0,
            'start_time': datetime.now()
        }
        
        # Display configuration
        self.display_info = {
            'window_name': 'Unsople - AI Waste Classification',
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'font_scale': 0.6,
            'font_color': (255, 255, 255),
            'bg_color': (0, 0, 0),
            'highlight_color': (0, 255, 0),
            'warning_color': (0, 0, 255)
        }
        
        self.logger.info("🚀 Unsople Inference System initialized")
        
    def setup_camera(self, camera_id=0):
        """
        Initialize camera for real-time inference
        
        Args:
            camera_id (int): Camera device ID
            
        Returns:
            cv2.VideoCapture: Camera object
        """
        self.logger.info(f"📷 Initializing camera (ID: {camera_id})")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            self.logger.error("❌ Failed to open camera")
            raise RuntimeError("Cannot open camera")
            
        self.logger.info("✅ Camera initialized successfully")
        return cap
    
    def draw_detection_info(self, frame, prediction, confidence, impact_data):
        """
        Draw detection information and impact data on frame
        
        Args:
            frame: OpenCV frame
            prediction (str): Predicted waste class
            confidence (float): Prediction confidence
            impact_data (dict): CO2 impact information
        """
        # Get display settings
        font = self.display_info['font']
        font_scale = self.display_info['font_scale']
        font_color = self.display_info['font_color']
        bg_color = self.display_info['bg_color']
        highlight_color = self.display_info['highlight_color']
        
        # Define positions and sizes
        y_offset = 30
        line_height = 25
        padding = 10
        
        # Information lines to display
        lines = [
            f"WASTE: {prediction.upper()}",
            f"CONFIDENCE: {confidence:.1%}",
            f"CO2 SAVED: {impact_data['co2_saved']:.3f} kg",
            f"TOTAL SAVED: {self.stats['total_co2_saved']:.3f} kg",
            f"ITEMS: {self.stats['total_predictions']}",
            f"BIN: {impact_data['bin_recommendation']}"
        ]
        
        # Calculate background rectangle size
        text_sizes = [cv2.getTextSize(line, font, font_scale, 2)[0] for line in lines]
        max_width = max(size[0] for size in text_sizes)
        total_height = len(lines) * line_height + padding * 2
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (padding, padding), 
                     (max_width + padding * 2, total_height), 
                     bg_color, -1)
        
        # Draw border
        cv2.rectangle(frame, 
                     (padding, padding), 
                     (max_width + padding * 2, total_height), 
                     highlight_color, 2)
        
        # Draw text lines
        for i, line in enumerate(lines):
            y_pos = padding + (i + 1) * line_height
            color = highlight_color if i in [0, 2, 3] else font_color
            cv2.putText(frame, line, (padding * 2, y_pos), 
                       font, font_scale, color, 2)
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = frame.shape[1] - bar_width - 20
        bar_y = 30
        
        # Background bar
        cv2.rectangle(frame, 
                     (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Confidence level
        confidence_width = int(bar_width * confidence)
        confidence_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
        
        cv2.rectangle(frame, 
                     (bar_x, bar_y), 
                     (bar_x + confidence_width, bar_y + bar_height), 
                     confidence_color, -1)
        
        # Confidence text
        confidence_text = f"Confidence: {confidence:.1%}"
        cv2.putText(frame, confidence_text, (bar_x, bar_y - 5), 
                   font, font_scale, font_color, 2)
    
    def process_frame(self, frame, save_detection=True):
        """
        Process single frame for waste classification
        
        Args:
            frame: OpenCV frame
            save_detection (bool): Whether to save detection results
            
        Returns:
            tuple: (processed_frame, prediction_data)
        """
        # Perform classification
        prediction, confidence, all_probabilities = self.classifier.predict(frame)
        
        if prediction:
            # Calculate CO2 impact
            impact_data = self.impact_calculator.calculate_impact(prediction)
            
            # Update statistics
            self.stats['total_predictions'] += 1
            self.stats['total_co2_saved'] += impact_data['co2_saved']
            
            # Draw information on frame
            self.draw_detection_info(frame, prediction, confidence, impact_data)
            
            # Save detection results
            if save_detection:
                self._save_detection_result(prediction, confidence, impact_data, frame)
            
            self.logger.info(f"🎯 Detection: {prediction} ({confidence:.1%}) - CO2 Saved: {impact_data['co2_saved']:.3f}kg")
            
            return frame, {
                'prediction': prediction,
                'confidence': confidence,
                'co2_saved': impact_data['co2_saved'],
                'total_co2_saved': self.stats['total_co2_saved'],
                'bin_recommendation': impact_data['bin_recommendation'],
                'timestamp': datetime.now().isoformat()
            }
        
        return frame, None
    
    def _save_detection_result(self, prediction, confidence, impact_data, frame):
        """
        Save detection result to CSV and optionally save image
        
        Args:
            prediction (str): Predicted class
            confidence (float): Confidence score
            impact_data (dict): CO2 impact data
            frame: Detected frame
        """
        timestamp = datetime.now()
        
        # Save to classification log
        classification_log_path = self.results_dir / "classification_log.csv"
        file_exists = classification_log_path.exists()
        
        with open(classification_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'prediction', 'confidence', 
                    'co2_saved_kg', 'total_co2_saved_kg', 'bin_recommendation'
                ])
            
            writer.writerow([
                timestamp.isoformat(),
                prediction,
                f"{confidence:.4f}",
                f"{impact_data['co2_saved']:.6f}",
                f"{self.stats['total_co2_saved']:.6f}",
                impact_data['bin_recommendation']
            ])
        
        # Save frame if confidence is high
        if confidence > 0.7:
            image_dir = self.results_dir / "detected_images"
            image_dir.mkdir(exist_ok=True)
            
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{prediction}_{confidence:.2f}.jpg"
            image_path = image_dir / filename
            cv2.imwrite(str(image_path), frame)
        
        # Update impact report (daily summary)
        self._update_impact_report(timestamp.date(), prediction, impact_data['co2_saved'])
    
    def _update_impact_report(self, date, prediction, co2_saved):
        """
        Update daily impact report
        
        Args:
            date: Date object
            prediction (str): Waste class
            co2_saved (float): CO2 saved amount
        """
        impact_report_path = self.results_dir / "impact_report.csv"
        
        # Read existing report or create new
        report_data = {}
        if impact_report_path.exists():
            with open(impact_report_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['date'] == date.isoformat():
                        report_data = row
                        break
        
        # Update data
        date_str = date.isoformat()
        if not report_data:
            report_data = {
                'date': date_str,
                'total_items': '0',
                'total_co2_saved_kg': '0.0',
                'plastic_items': '0', 'plastic_co2_kg': '0.0',
                'paper_items': '0', 'paper_co2_kg': '0.0',
                'organic_items': '0', 'organic_co2_kg': '0.0',
                'metal_items': '0', 'metal_co2_kg': '0.0',
                'glass_items': '0', 'glass_co2_kg': '0.0'
            }
        
        # Update counts and CO2
        report_data['total_items'] = str(int(report_data['total_items']) + 1)
        report_data['total_co2_saved_kg'] = str(float(report_data['total_co2_saved_kg']) + co2_saved)
        
        prediction_field = f"{prediction}_items"
        co2_field = f"{prediction}_co2_kg"
        
        if prediction_field in report_data:
            report_data[prediction_field] = str(int(report_data[prediction_field]) + 1)
            report_data[co2_field] = str(float(report_data[co2_field]) + co2_saved)
        
        # Write updated report
        fieldnames = [
            'date', 'total_items', 'total_co2_saved_kg',
            'plastic_items', 'plastic_co2_kg',
            'paper_items', 'paper_co2_kg', 
            'organic_items', 'organic_co2_kg',
            'metal_items', 'metal_co2_kg',
            'glass_items', 'glass_co2_kg'
        ]
        
        # Read all rows and update
        all_rows = []
        if impact_report_path.exists():
            with open(impact_report_path, 'r') as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)
        
        # Update or append row
        updated = False
        for i, row in enumerate(all_rows):
            if row['date'] == date_str:
                all_rows[i] = report_data
                updated = True
                break
        
        if not updated:
            all_rows.append(report_data)
        
        # Write back
        with open(impact_report_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    
    def run_realtime_inference(self, camera_id=0, save_output=False):
        """
        Run real-time inference from camera
        
        Args:
            camera_id (int): Camera device ID
            save_output (bool): Whether to save output video
        """
        self.logger.info("🎥 Starting real-time inference...")
        
        cap = self.setup_camera(camera_id)
        video_writer = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("❌ Failed to capture frame")
                    break
                
                # Process frame
                processed_frame, detection_data = self.process_frame(frame)
                
                # Initialize video writer if saving output
                if save_output and video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    output_path = self.results_dir / "inference_output.avi"
                    video_writer = cv2.VideoWriter(
                        str(output_path), fourcc, 20.0, 
                        (processed_frame.shape[1], processed_frame.shape[0])
                    )
                
                # Write frame if saving
                if video_writer is not None:
                    video_writer.write(processed_frame)
                
                # Display frame
                cv2.imshow(self.display_info['window_name'], processed_frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.logger.info("⏹️ Stopping inference by user request")
                    break
                elif key == ord('s'):  # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = self.results_dir / f"manual_capture_{timestamp}.jpg"
                    cv2.imwrite(str(save_path), processed_frame)
                    self.logger.info(f"💾 Manual capture saved: {save_path}")
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Inference interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Inference error: {e}")
        finally:
            # Cleanup
            if video_writer is not None:
                video_writer.release()
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_final_stats()
    
    def process_image_file(self, image_path, display=True):
        """
        Process single image file
        
        Args:
            image_path (str): Path to image file
            display (bool): Whether to display result
            
        Returns:
            dict: Detection results
        """
        self.logger.info(f"📷 Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            self.logger.error(f"❌ Image file not found: {image_path}")
            return None
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            self.logger.error(f"❌ Failed to read image: {image_path}")
            return None
        
        # Process image
        processed_frame, detection_data = self.process_frame(frame, save_detection=True)
        
        if display and detection_data:
            # Display result
            cv2.imshow('Unsople - Image Result', processed_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detection_data
    
    def _print_final_stats(self):
        """Print final inference statistics"""
        duration = datetime.now() - self.stats['start_time']
        
        print("\n" + "="*50)
        print("🎉 UNSOPLE INFERENCE SUMMARY")
        print("="*50)
        print(f"📊 Total Predictions: {self.stats['total_predictions']}")
        print(f"🌱 Total CO2 Saved: {self.stats['total_co2_saved']:.3f} kg")
        print(f"⏱️  Duration: {duration}")
        print(f"📁 Results saved to: {self.results_dir}")
        print("="*50)

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Unsople AI Waste Classification System')
    parser.add_argument('--mode', type=str, choices=['camera', 'image'], default='camera',
                       help='Inference mode: camera (real-time) or image (single file)')
    parser.add_argument('--image', type=str, help='Path to image file for image mode')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--save-video', action='store_true', help='Save output video in camera mode')
    
    args = parser.parse_args()
    
    try:
        # Initialize Unsople inference system
        unsople = UnsopleInference(args.config)
        
        if args.mode == 'camera':
            # Real-time camera inference
            unsople.run_realtime_inference(
                camera_id=args.camera,
                save_output=args.save_video
            )
        elif args.mode == 'image':
            # Single image inference
            if not args.image:
                print("❌ Please provide image path with --image argument")
                return
            
            result = unsople.process_image_file(args.image, display=True)
            if result:
                print(f"✅ Detection Result: {result}")
            else:
                print("❌ No detection result")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()