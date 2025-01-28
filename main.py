from utils import AugmentedBenchmarkDataset
import argparse
import logging
import yaml

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def check_save_config(config_path: str) -> bool:
    """
    Check if config contains save dataset information
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return 'save_dataset' in config and 'repo_dataset_name' in config['save_dataset']

def main():
    parser = argparse.ArgumentParser(description='Create augmented speech benchmark dataset')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration YAML file'
    )

    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting dataset creation process")
        
        # Create dataset
        dataset = AugmentedBenchmarkDataset(args.config)
        
        # Check if we need to push to HF
        if check_save_config(args.config):
            logger.info("Pushing dataset to HuggingFace Hub")
            dataset.push_HF()
        else:
            logger.info("No HuggingFace Hub configuration found. Skipping upload.")
        
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
