import argparse
import os
from components.trainloop import TrainLoop
from data.generate_sample_data import main as generate_data


def main():
    """
    Main function to demonstrate the dependency injection system.
    """
    parser = argparse.ArgumentParser(description='ML Dependency Injection')
    parser.add_argument('--config', type=str, default='configs/sample_config.yaml',
                        help='Path to the YAML configuration file')
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate sample data before running')
    
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if args.generate_data:
        print("Generating sample data...")
        generate_data()
    
    config_path = args.config
    print(f"Using configuration from: {config_path}")
    
    train_loop = TrainLoop.create(config_path)
    
    print("\nStarting training loop...")
    train_loop.execute()
    print("\nTraining completed.")


if __name__ == "__main__":
    main() 