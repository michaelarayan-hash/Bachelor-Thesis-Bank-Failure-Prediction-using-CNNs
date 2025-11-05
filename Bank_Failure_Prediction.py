"""
Master Pipeline Script for Bank Failure Prediction

This script orchestrates the entire pipeline from preprocessing to final evaluation.
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from typing import List


class PipelineRunner:
    """Orchestrates the complete bank failure prediction pipeline."""
    
    def __init__(self, config: dict):
        """
        Initialize pipeline runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.start_time = datetime.now()
        self.log_file = os.path.join(config['log_dir'], f"pipeline_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log")
        os.makedirs(config['log_dir'], exist_ok=True)
    
    def log(self, message: str):
        """Log message to console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def run_command(self, cmd: List[str], step_name: str) -> bool:
        """
        Run a command and handle errors.
        
        Args:
            cmd: Command as list of strings
            step_name: Name of the pipeline step
            
        Returns:
            True if successful, False otherwise
        """
        self.log(f"\n{'='*60}")
        self.log(f"Starting: {step_name}")
        self.log(f"Command: {' '.join(cmd)}")
        self.log(f"{'='*60}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            self.log(result.stdout)
            self.log(f"✓ {step_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"✗ {step_name} failed with error:")
            self.log(e.stdout)
            return False
    
    def step1_preprocess(self) -> bool:
        """Run preprocessing step."""
        config = self.config
        
        cmd = [
            sys.executable, 'preprocessing.py',
            '--panel-path', config['panel_path'],
            '--cbr-path', config['cbr_path'],
            '--output-dir', config['preprocess_output'],
            '--time-span', str(config['time_span'])
        ]
        
        if config['variable_sets']:
            cmd.extend(['--variable-sets'] + config['variable_sets'])
        
        return self.run_command(cmd, "Preprocessing")
    
    def step2_initial_training(self) -> bool:
        """Run initial training step."""
        config = self.config
        
        input_dir = os.path.join(config['preprocess_output'], f"Banks_{config['time_span'] * 2}")
        
        cmd = [
            sys.executable, 'train_initial.py',
            '--input-dir', input_dir,
            '--output-dir', config['initial_training_output'],
            '--epochs', str(config['initial_epochs']),
            '--batch-size', str(config['batch_size'])
        ]
        
        if config.get('gpu') is not None:
            cmd.extend(['--gpu', str(config['gpu'])])
        
        return self.run_command(cmd, "Initial Training")
    
    def step3_final_training(self) -> bool:
        """Run final training step."""
        config = self.config
        
        input_dir = os.path.join(config['preprocess_output'], f"Banks_{config['time_span'] * 2}")
        epochs_config = os.path.join(config['initial_training_output'], 'best_epochs.csv')
        
        cmd = [
            sys.executable, 'train_final.py',
            '--input-dir', input_dir,
            '--output-dir', config['final_training_output'],
            '--batch-size', str(config['batch_size']),
            '--default-epochs', str(config['default_final_epochs'])
        ]
        
        if os.path.exists(epochs_config):
            cmd.extend(['--epochs-config', epochs_config])
        
        if config.get('gpu') is not None:
            cmd.extend(['--gpu', str(config['gpu'])])
        
        return self.run_command(cmd, "Final Training and Evaluation")
    
    def run(self, skip_steps: List[str] = None):
        """
        Run the complete pipeline.
        
        Args:
            skip_steps: List of steps to skip ('preprocess', 'initial', 'final')
        """
        skip_steps = skip_steps or []
        
        self.log("="*60)
        self.log("BANK FAILURE PREDICTION PIPELINE")
        self.log("="*60)
        self.log(f"Start time: {self.start_time}")
        self.log(f"\nConfiguration:")
        for key, value in self.config.items():
            self.log(f"  {key}: {value}")
        
        success = True
        
        # Step 1: Preprocessing
        if 'preprocess' not in skip_steps:
            if not self.step1_preprocess():
                self.log("\n✗ Pipeline failed at preprocessing step")
                return False
        else:
            self.log("\nSkipping preprocessing step")
        
        # Step 2: Initial Training
        if 'initial' not in skip_steps:
            if not self.step2_initial_training():
                self.log("\n✗ Pipeline failed at initial training step")
                return False
        else:
            self.log("\nSkipping initial training step")
        
        # Step 3: Final Training
        if 'final' not in skip_steps:
            if not self.step3_final_training():
                self.log("\n✗ Pipeline failed at final training step")
                return False
        else:
            self.log("\nSkipping final training step")
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.log("\n" + "="*60)
        self.log("PIPELINE COMPLETED SUCCESSFULLY")
        self.log("="*60)
        self.log(f"End time: {end_time}")
        self.log(f"Total duration: {duration}")
        self.log(f"\nResults saved to:")
        self.log(f"  Preprocessed data: {self.config['preprocess_output']}")
        self.log(f"  Initial training: {self.config['initial_training_output']}")
        self.log(f"  Final results: {self.config['final_training_output']}")
        self.log(f"\nLog file: {self.log_file}")
        
        return True


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def create_default_config(output_file: str):
    """Create a default configuration file."""
    default_config = {
        "panel_path": "data/panel.dta",
        "cbr_path": "data/cbrdataT.dta",
        "preprocess_output": "output",
        "initial_training_output": "plots",
        "final_training_output": "results",
        "log_dir": "logs",
        "time_span": 6,
        "initial_epochs": 2500,
        "default_final_epochs": 2500,
        "batch_size": 64,
        "variable_sets": ["all"],
        "gpu": None
    }
    
    with open(output_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default configuration saved to: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run complete bank failure prediction pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/pipeline_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--create-config',
        type=str,
        default=None,
        help='Create default config file at specified path'
    )
    parser.add_argument(
        '--skip',
        nargs='+',
        choices=['preprocess', 'initial', 'final'],
        default=[],
        help='Steps to skip'
    )
    parser.add_argument(
        '--panel-path',
        type=str,
        help='Override panel.dta path'
    )
    parser.add_argument(
        '--cbr-path',
        type=str,
        help='Override cbrdataT.dta path'
    )
    parser.add_argument(
        '--time-span',
        type=int,
        help='Override time span'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        help='Override GPU device ID'
    )
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        print(f"Config file not found: {args.config}")
        print("Creating default configuration...")
        os.makedirs(os.path.dirname(args.config) or '.', exist_ok=True)
        create_default_config(args.config)
        config = load_config(args.config)
    
    # Override with command line arguments
    if args.panel_path:
        config['panel_path'] = args.panel_path
    if args.cbr_path:
        config['cbr_path'] = args.cbr_path
    if args.time_span:
        config['time_span'] = args.time_span
    if args.gpu is not None:
        config['gpu'] = args.gpu
    
    # Validate required files
    if 'preprocess' not in args.skip:
        if not os.path.exists(config['panel_path']):
            print(f"Error: Panel file not found: {config['panel_path']}")
            return
        if not os.path.exists(config['cbr_path']):
            print(f"Error: CBR file not found: {config['cbr_path']}")
            return
    
    # Run pipeline
    runner = PipelineRunner(config)
    success = runner.run(skip_steps=args.skip)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()