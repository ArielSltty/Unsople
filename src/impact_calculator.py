# impact_calculator.py
"""
🌱 Unsople - AI-Powered Smart Sorting System
📍 CO2 Impact Calculation Module
🎯 Calculate environmental impact of proper waste sorting
"""

import json
import csv
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

class WasteCategory(Enum):
    """Enumeration of waste categories"""
    PLASTIC = "plastic"
    PAPER = "paper" 
    ORGANIC = "organic"
    METAL = "metal"
    GLASS = "glass"

@dataclass
class CO2ImpactResult:
    """Data class to store CO2 impact calculation results"""
    waste_type: str
    co2_saved_kg: float
    bin_recommendation: str
    equivalent_trees: float
    equivalent_cars: float
    weight_kg: float
    timestamp: datetime

class CO2ImpactCalculator:
    """
    Calculator for CO2 emission savings from proper waste sorting
    Based on scientific research and environmental impact studies
    """
    
    # CO2 savings per kg of material (kg CO2 equivalent per kg of waste)
    # Sources: EPA, World Bank, and environmental research data
    CO2_SAVINGS_PER_KG = {
        WasteCategory.PLASTIC.value: 6.0,    # Based on recycling vs virgin plastic production
        WasteCategory.PAPER.value: 3.5,      # Based on recycling vs new paper production  
        WasteCategory.ORGANIC.value: 1.4,    # Based on composting vs landfill methane
        WasteCategory.METAL.value: 9.0,      # Based on aluminum recycling energy savings
        WasteCategory.GLASS.value: 0.3       # Based on recycling vs new glass production
    }
    
    # Average weight estimates for common waste items (kg)
    # These are typical weights for standard items
    AVERAGE_WEIGHTS = {
        WasteCategory.PLASTIC.value: {
            'bottle': 0.025,
            'container': 0.015,
            'wrapper': 0.005,
            'bag': 0.010,
            'default': 0.015
        },
        WasteCategory.PAPER.value: {
            'newspaper': 0.150,
            'cardboard': 0.200,
            'office_paper': 0.005,
            'box': 0.300,
            'default': 0.050
        },
        WasteCategory.ORGANIC.value: {
            'food_waste': 0.100,
            'fruit': 0.150,
            'vegetable': 0.120,
            'compost': 0.200,
            'default': 0.120
        },
        WasteCategory.METAL.value: {
            'can': 0.015,
            'foil': 0.002,
            'container': 0.030,
            'utensils': 0.020,
            'default': 0.015
        },
        WasteCategory.GLASS.value: {
            'bottle': 0.400,
            'jar': 0.300,
            'container': 0.250,
            'broken_glass': 0.100,
            'default': 0.300
        }
    }
    
    # Bin color recommendations based on international standards
    BIN_RECOMMENDATIONS = {
        WasteCategory.PLASTIC.value: "Yellow Bin (Recycling)",
        WasteCategory.PAPER.value: "Blue Bin (Recycling)", 
        WasteCategory.ORGANIC.value: "Green Bin (Compost)",
        WasteCategory.METAL.value: "Gray Bin (Recycling)",
        WasteCategory.GLASS.value: "Brown Bin (Recycling)"
    }
    
    # Environmental equivalents for impact visualization
    ENVIRONMENTAL_EQUIVALENTS = {
        'tree_absorption_kg_co2_per_year': 21.77,  # Average tree absorbs ~21.77 kg CO2 per year
        'car_emissions_kg_co2_per_km': 0.12,       # Average car emits ~0.12 kg CO2 per km
        'smartphone_charges_per_kg_co2': 1200,      # ~1200 smartphone charges = 1 kg CO2
        'lightbulb_hours_per_kg_co2': 100           # ~100 hours of LED light = 1 kg CO2
    }

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize CO2 Impact Calculator
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.config = self._load_config()
        
        # Initialize impact tracking
        self.daily_impact = {}
        self.session_impact = {
            'total_items': 0,
            'total_co2_saved_kg': 0.0,
            'category_breakdown': {category: 0 for category in self.CO2_SAVINGS_PER_KG.keys()}
        }
        
        self.logger.info("✅ CO2 Impact Calculator initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the calculator"""
        logger = logging.getLogger('CO2ImpactCalculator')
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
            "impact_calculation": {
                "use_dynamic_weights": True,
                "default_weight_kg": 0.015,
                "conservative_estimates": True,
                "include_methane_factor": True
            },
            "reporting": {
                "auto_save_reports": True,
                "report_directory": "results",
                "daily_summary": True
            },
            "environmental_equivalents": self.ENVIRONMENTAL_EQUIVALENTS
        }

    def calculate_impact(self, waste_type: str, item_type: Optional[str] = None, 
                        custom_weight: Optional[float] = None) -> Dict:
        """
        Calculate CO2 impact for a waste item
        
        Args:
            waste_type (str): Type of waste (plastic, paper, organic, metal, glass)
            item_type (str, optional): Specific item type for weight estimation
            custom_weight (float, optional): Custom weight in kg
            
        Returns:
            Dict: Comprehensive impact calculation results
        """
        try:
            # Validate waste type
            if waste_type.lower() not in self.CO2_SAVINGS_PER_KG:
                self.logger.warning(f"⚠️ Unknown waste type: {waste_type}, using default")
                waste_type = WasteCategory.PLASTIC.value

            # Calculate weight
            weight_kg = self._calculate_weight(waste_type, item_type, custom_weight)
            
            # Calculate CO2 savings
            co2_saved = self._calculate_co2_savings(waste_type, weight_kg)
            
            # Get bin recommendation
            bin_recommendation = self.BIN_RECOMMENDATIONS.get(waste_type, "General Waste")
            
            # Calculate environmental equivalents
            equivalents = self._calculate_environmental_equivalents(co2_saved)
            
            # Create result object
            impact_result = CO2ImpactResult(
                waste_type=waste_type,
                co2_saved_kg=co2_saved,
                bin_recommendation=bin_recommendation,
                equivalent_trees=equivalents['trees'],
                equivalent_cars=equivalents['car_km'],
                weight_kg=weight_kg,
                timestamp=datetime.now()
            )
            
            # Update session tracking
            self._update_session_tracking(impact_result)
            
            # Update daily tracking
            self._update_daily_tracking(impact_result)
            
            # Auto-save reports if enabled
            if self.config['reporting']['auto_save_reports']:
                self._auto_save_detection(impact_result)
            
            self.logger.info(f"🌱 Impact calculated: {waste_type} - {co2_saved:.3f} kg CO2 saved")
            
            return self._format_result(impact_result, equivalents)
            
        except Exception as e:
            self.logger.error(f"❌ Impact calculation failed: {e}")
            return self._get_error_result()

    def _calculate_weight(self, waste_type: str, item_type: Optional[str], 
                         custom_weight: Optional[float]) -> float:
        """
        Calculate item weight based on type and configuration
        
        Args:
            waste_type (str): Waste category
            item_type (str): Specific item type
            custom_weight (float): Custom weight if provided
            
        Returns:
            float: Weight in kg
        """
        if custom_weight is not None:
            return max(0.001, custom_weight)  # Minimum 1g
        
        if (self.config['impact_calculation']['use_dynamic_weights'] and 
            item_type and waste_type in self.AVERAGE_WEIGHTS):
            
            weight = self.AVERAGE_WEIGHTS[waste_type].get(
                item_type, 
                self.AVERAGE_WEIGHTS[waste_type]['default']
            )
            
            # Apply conservative factor if enabled
            if self.config['impact_calculation']['conservative_estimates']:
                weight *= 0.8  # Use 80% of estimated weight
            
            return weight
        
        return self.config['impact_calculation']['default_weight_kg']

    def _calculate_co2_savings(self, waste_type: str, weight_kg: float) -> float:
        """
        Calculate CO2 savings based on waste type and weight
        
        Args:
            waste_type (str): Type of waste
            weight_kg (float): Weight in kilograms
            
        Returns:
            float: CO2 savings in kg
        """
        base_savings = self.CO2_SAVINGS_PER_KG[waste_type] * weight_kg
        
        # Apply methane factor for organic waste (landfill methane is potent GHG)
        if (waste_type == WasteCategory.ORGANIC.value and 
            self.config['impact_calculation']['include_methane_factor']):
            base_savings *= 1.25  # 25% additional savings due to methane avoidance
        
        return round(base_savings, 6)  # Round to avoid floating point issues

    def _calculate_environmental_equivalents(self, co2_saved: float) -> Dict:
        """
        Calculate environmental equivalents for impact visualization
        
        Args:
            co2_saved (float): CO2 savings in kg
            
        Returns:
            Dict: Various environmental equivalents
        """
        return {
            'trees': co2_saved / self.ENVIRONMENTAL_EQUIVALENTS['tree_absorption_kg_co2_per_year'],
            'car_km': co2_saved / self.ENVIRONMENTAL_EQUIVALENTS['car_emissions_kg_co2_per_km'],
            'smartphone_charges': co2_saved * self.ENVIRONMENTAL_EQUIVALENTS['smartphone_charges_per_kg_co2'],
            'lightbulb_hours': co2_saved * self.ENVIRONMENTAL_EQUIVALENTS['lightbulb_hours_per_kg_co2']
        }

    def _update_session_tracking(self, impact_result: CO2ImpactResult):
        """Update session-level impact tracking"""
        self.session_impact['total_items'] += 1
        self.session_impact['total_co2_saved_kg'] += impact_result.co2_saved_kg
        self.session_impact['category_breakdown'][impact_result.waste_type] += 1

    def _update_daily_tracking(self, impact_result: CO2ImpactResult):
        """Update daily impact tracking"""
        today = date.today().isoformat()
        
        if today not in self.daily_impact:
            self.daily_impact[today] = {
                'date': today,
                'total_items': 0,
                'total_co2_saved_kg': 0.0,
                'category_items': {category: 0 for category in self.CO2_SAVINGS_PER_KG.keys()},
                'category_co2': {category: 0.0 for category in self.CO2_SAVINGS_PER_KG.keys()}
            }
        
        daily = self.daily_impact[today]
        daily['total_items'] += 1
        daily['total_co2_saved_kg'] += impact_result.co2_saved_kg
        daily['category_items'][impact_result.waste_type] += 1
        daily['category_co2'][impact_result.waste_type] += impact_result.co2_saved_kg

    def _auto_save_detection(self, impact_result: CO2ImpactResult):
        """Automatically save detection to impact report"""
        try:
            report_dir = Path(self.config['reporting']['report_directory'])
            report_dir.mkdir(exist_ok=True)
            
            impact_report_path = report_dir / "impact_report.csv"
            
            # Prepare data for CSV
            data = {
                'timestamp': impact_result.timestamp.isoformat(),
                'waste_type': impact_result.waste_type,
                'co2_saved_kg': f"{impact_result.co2_saved_kg:.6f}",
                'weight_kg': f"{impact_result.weight_kg:.4f}",
                'bin_recommendation': impact_result.bin_recommendation,
                'equivalent_trees': f"{impact_result.equivalent_trees:.4f}",
                'equivalent_car_km': f"{impact_result.equivalent_cars:.2f}"
            }
            
            # Write to CSV
            file_exists = impact_report_path.exists()
            with open(impact_report_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not auto-save impact report: {e}")

    def _format_result(self, impact_result: CO2ImpactResult, equivalents: Dict) -> Dict:
        """
        Format impact result for API response
        
        Args:
            impact_result (CO2ImpactResult): Raw impact result
            equivalents (Dict): Environmental equivalents
            
        Returns:
            Dict: Formatted result dictionary
        """
        return {
            'success': True,
            'waste_type': impact_result.waste_type,
            'co2_saved_kg': impact_result.co2_saved_kg,
            'weight_kg': impact_result.weight_kg,
            'bin_recommendation': impact_result.bin_recommendation,
            'environmental_equivalents': {
                'equivalent_trees': round(impact_result.equivalent_trees, 4),
                'equivalent_car_km': round(impact_result.equivalent_cars, 2),
                'smartphone_charges': int(equivalents['smartphone_charges']),
                'led_lightbulb_hours': int(equivalents['lightbulb_hours'])
            },
            'timestamp': impact_result.timestamp.isoformat(),
            'session_total_co2_kg': self.session_impact['total_co2_saved_kg'],
            'session_total_items': self.session_impact['total_items']
        }

    def _get_error_result(self) -> Dict:
        """Get error result dictionary"""
        return {
            'success': False,
            'waste_type': 'unknown',
            'co2_saved_kg': 0.0,
            'weight_kg': 0.0,
            'bin_recommendation': 'General Waste',
            'environmental_equivalents': {},
            'error': 'Calculation failed'
        }

    def get_session_summary(self) -> Dict:
        """
        Get current session impact summary
        
        Returns:
            Dict: Session summary statistics
        """
        return {
            'session_duration': 'current',  # Could be enhanced with start time tracking
            'total_items_processed': self.session_impact['total_items'],
            'total_co2_saved_kg': round(self.session_impact['total_co2_saved_kg'], 3),
            'category_breakdown': self.session_impact['category_breakdown'],
            'average_co2_per_item': (
                self.session_impact['total_co2_saved_kg'] / 
                max(1, self.session_impact['total_items'])
            )
        }

    def get_daily_summary(self, target_date: Optional[date] = None) -> Dict:
        """
        Get daily impact summary
        
        Args:
            target_date (date, optional): Specific date, defaults to today
            
        Returns:
            Dict: Daily summary statistics
        """
        if target_date is None:
            target_date = date.today()
        
        date_str = target_date.isoformat()
        
        if date_str not in self.daily_impact:
            return {
                'date': date_str,
                'total_items': 0,
                'total_co2_saved_kg': 0.0,
                'category_breakdown': {},
                'message': 'No data for this date'
            }
        
        daily_data = self.daily_impact[date_str]
        
        return {
            'date': date_str,
            'total_items': daily_data['total_items'],
            'total_co2_saved_kg': round(daily_data['total_co2_saved_kg'], 3),
            'category_breakdown': {
                category: {
                    'items': daily_data['category_items'][category],
                    'co2_saved_kg': round(daily_data['category_co2'][category], 3)
                }
                for category in self.CO2_SAVINGS_PER_KG.keys()
            },
            'most_common_category': max(
                daily_data['category_items'].items(), 
                key=lambda x: x[1]
            )[0] if daily_data['total_items'] > 0 else 'none'
        }

    def generate_impact_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive impact report
        
        Args:
            output_path (str, optional): Custom output path
            
        Returns:
            str: Path to generated report
        """
        try:
            if output_path is None:
                report_dir = Path(self.config['reporting']['report_directory'])
                report_dir.mkdir(exist_ok=True)
                output_path = report_dir / f"unsople_impact_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            else:
                output_path = Path(output_path)
            
            # Prepare report data
            report_data = []
            for date_str, daily_data in self.daily_impact.items():
                row = {
                    'date': date_str,
                    'total_items': daily_data['total_items'],
                    'total_co2_saved_kg': daily_data['total_co2_saved_kg']
                }
                
                # Add category-specific data
                for category in self.CO2_SAVINGS_PER_KG.keys():
                    row[f'{category}_items'] = daily_data['category_items'][category]
                    row[f'{category}_co2_kg'] = daily_data['category_co2'][category]
                
                report_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(report_data)
            
            # Add summary row
            if not df.empty:
                summary_row = {
                    'date': 'TOTAL',
                    'total_items': df['total_items'].sum(),
                    'total_co2_saved_kg': df['total_co2_saved_kg'].sum()
                }
                
                for category in self.CO2_SAVINGS_PER_KG.keys():
                    summary_row[f'{category}_items'] = df[f'{category}_items'].sum()
                    summary_row[f'{category}_co2_kg'] = df[f'{category}_co2_kg'].sum()
                
                df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
            
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"📊 Impact report generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate impact report: {e}")
            raise

    def reset_session(self):
        """Reset current session tracking"""
        self.session_impact = {
            'total_items': 0,
            'total_co2_saved_kg': 0.0,
            'category_breakdown': {category: 0 for category in self.CO2_SAVINGS_PER_KG.keys()}
        }
        self.logger.info("🔄 Session tracking reset")

    def get_environmental_impact_story(self, co2_saved: float) -> Dict:
        """
        Create environmental impact story for visualization
        
        Args:
            co2_saved (float): Total CO2 saved in kg
            
        Returns:
            Dict: Impact story with multiple perspectives
        """
        equivalents = self._calculate_environmental_equivalents(co2_saved)
        
        return {
            'co2_saved_kg': co2_saved,
            'stories': [
                {
                    'title': 'Tree Planting Impact',
                    'description': f"This is equivalent to the CO2 absorbed by {equivalents['trees']:.1f} trees in one year",
                    'icon': '🌳'
                },
                {
                    'title': 'Car Emission Savings', 
                    'description': f"Equal to not driving {equivalents['car_km']:.1f} km in a typical car",
                    'icon': '🚗'
                },
                {
                    'title': 'Energy Savings',
                    'description': f"Enough to charge {equivalents['smartphone_charges']:,} smartphones",
                    'icon': '🔋'
                },
                {
                    'title': 'Lighting Impact',
                    'description': f"Could power an LED bulb for {equivalents['lightbulb_hours']:,} hours",
                    'icon': '💡'
                }
            ]
        }

# Utility functions for standalone use
def create_impact_calculator(config_path: str = "config.json") -> CO2ImpactCalculator:
    """
    Factory function to create CO2ImpactCalculator instance
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        CO2ImpactCalculator: Calculator instance
    """
    return CO2ImpactCalculator(config_path)

def calculate_single_impact(waste_type: str, weight_kg: float = None, 
                          config_path: str = "config.json") -> Dict:
    """
    Convenience function for single impact calculation
    
    Args:
        waste_type (str): Type of waste
        weight_kg (float, optional): Custom weight in kg
        config_path (str): Path to config file
        
    Returns:
        Dict: Impact calculation results
    """
    calculator = create_impact_calculator(config_path)
    return calculator.calculate_impact(waste_type, custom_weight=weight_kg)

if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Unsople CO2 Impact Calculator Test')
    parser.add_argument('--waste-type', type=str, required=True, 
                       choices=['plastic', 'paper', 'organic', 'metal', 'glass'],
                       help='Type of waste material')
    parser.add_argument('--weight', type=float, help='Weight in kg (optional)')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    
    args = parser.parse_args()
    
    try:
        result = calculate_single_impact(args.waste_type, args.weight, args.config)
        
        print("\n" + "="*60)
        print("🌱 UNSOPLE CO2 IMPACT CALCULATION")
        print("="*60)
        
        if result['success']:
            print(f"✅ Waste Type: {result['waste_type'].upper()}")
            print(f"📊 CO2 Saved: {result['co2_saved_kg']:.3f} kg")
            print(f"⚖️  Estimated Weight: {result['weight_kg']:.3f} kg")
            print(f"🗑️  Bin Recommendation: {result['bin_recommendation']}")
            print("\n🌍 Environmental Equivalents:")
            equivalents = result['environmental_equivalents']
            print(f"   🌳 {equivalents['equivalent_trees']:.2f} trees absorbing CO2 for one year")
            print(f"   🚗 {equivalents['equivalent_car_km']:.1f} km not driven")
            print(f"   📱 {equivalents['smartphone_charges']:,} smartphone charges")
            print(f"   💡 {equivalents['led_lightbulb_hours']:,} hours of LED lighting")
        else:
            print("❌ Calculation failed")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")