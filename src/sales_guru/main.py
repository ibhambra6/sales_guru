#!/usr/bin/env python
"""Main entry point for Sales Guru application."""

import sys
import warnings
import os
import argparse
from typing import Dict, Any
from pathlib import Path

from sales_guru.crew import SalesGuru
from sales_guru.error_handling import with_retry, handle_exceptions
from sales_guru.config import config
from sales_guru.logging_config import setup_production_logging, setup_development_logging, get_logger

# Setup logging based on environment
if os.getenv('SALES_GURU_ENV') == 'production':
    setup_production_logging()
else:
    setup_development_logging()

logger = get_logger("SalesGuruMain")

# Suppress warnings from pysbd
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.


def get_default_inputs() -> Dict[str, Any]:
    """Get default input parameters for the crew.
    
    Returns:
        Dictionary containing default input parameters
    """
    return {
        'file_name': config.default_csv_path,
        'company_name': 'Oceaneering Mobile Robotics',
        'company_description': (
            'Oceaneering Mobile Robotics specializes in autonomous material handling solutions. '
            'Their product line includes an autonomous forklift that transports racks and pallets, '
            'and an autonomous under carriage vehicle designed for smaller loads that need to be '
            'raised off the ground.'
        )
    }


def validate_csv_file(csv_path: str) -> str:
    """Validate that the CSV file exists and is readable.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Validated CSV file path
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the file is not a CSV file
    """
    path = Path(csv_path)
    
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {csv_path}")
    
    if path.suffix.lower() != '.csv':
        raise ValueError(f"File must be a CSV file (*.csv): {csv_path}")
    
    return str(path.resolve())


@with_retry(max_retries=5)
@handle_exceptions
def run(csv_file: str = None, company_name: str = None, company_description: str = None) -> None:
    """Run the crew with task completion guarantees.
    
    Args:
        csv_file: Path to CSV file (if None, will use default)
        company_name: Company name (if None, will use default)
        company_description: Company description (if None, will use default)
    """
    # Use defaults if not provided
    if csv_file is None:
        csv_file = config.default_csv_path
    
    if company_name is None:
        company_name = "Oceaneering Mobile Robotics"
    
    if company_description is None:
        company_description = get_default_inputs()['company_description']
    
    # Validate CSV file
    try:
        csv_file = validate_csv_file(csv_file)
        logger.info(f"Using CSV file: {csv_file}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"CSV file validation failed: {e}")
        raise
    
    inputs = {
        'file_name': csv_file,
        'company_name': company_name,
        'company_description': company_description
    }
    
    logger.info(f"Starting SalesGuru execution with CSV file: {csv_file}")
    logger.info(f"Company: {company_name}")
    SalesGuru(csv_file_path=csv_file).kickoff(inputs=inputs)
    logger.info("SalesGuru execution completed successfully")


@with_retry(max_retries=3)
@handle_exceptions
def train() -> None:
    """Train the crew with task completion guarantees."""
    # Use default values
    company_name = "Oceaneering Mobile Robotics"
    company_description = get_default_inputs()['company_description']
    file_name = config.default_csv_path
    iterations = 1
    filename = None
    
    inputs = {
        'file_name': file_name,
        'company_name': company_name,
        'company_description': company_description
    }
    
    logger.info(f"Starting training with {iterations} iterations")
    SalesGuru().train(n_iterations=iterations, filename=filename, inputs=inputs)
    logger.info("Training completed successfully")


@with_retry(max_retries=3)
@handle_exceptions
def replay(task_id: str = None) -> None:
    """Replay the crew execution with task completion guarantees."""
    if not task_id:
        raise ValueError("Task ID is required for replay")
    
    logger.info(f"Starting replay for task ID: {task_id}")
    SalesGuru().replay(task_id=task_id)
    logger.info("Replay completed successfully")


@with_retry(max_retries=3)
@handle_exceptions
def test() -> None:
    """Test the crew execution with task completion guarantees."""
    # Use default values
    company_name = "Oceaneering Mobile Robotics"
    company_description = get_default_inputs()['company_description']
    file_name = config.default_csv_path
    iterations = 1
    model_name = None
    
    inputs = {
        'file_name': file_name,
        'company_name': company_name,
        'company_description': company_description
    }
    
    logger.info(f"Starting test with {iterations} iterations")
    SalesGuru().test(n_iterations=iterations, openai_model_name=model_name, inputs=inputs)
    logger.info("Test completed successfully")


def train_with_args(args) -> None:
    """Train the crew with command line arguments."""
    # Use provided arguments or defaults
    csv_file = args.csv or config.default_csv_path
    company_name = args.company or "Oceaneering Mobile Robotics"
    company_description = args.description or get_default_inputs()['company_description']
    
    # Validate CSV file
    try:
        csv_file = validate_csv_file(csv_file)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"CSV file validation failed: {e}")
        raise
    
    inputs = {
        'file_name': csv_file,
        'company_name': company_name,
        'company_description': company_description
    }
    
    logger.info(f"Starting training with {args.iterations} iterations")
    SalesGuru(csv_file_path=csv_file).train(
        n_iterations=args.iterations, 
        filename=args.output, 
        inputs=inputs
    )
    logger.info("Training completed successfully")


def test_with_args(args) -> None:
    """Test the crew with command line arguments."""
    # Use provided arguments or defaults
    csv_file = args.csv or config.default_csv_path
    company_name = args.company or "Oceaneering Mobile Robotics"
    company_description = args.description or get_default_inputs()['company_description']
    
    # Validate CSV file
    try:
        csv_file = validate_csv_file(csv_file)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"CSV file validation failed: {e}")
        raise
    
    inputs = {
        'file_name': csv_file,
        'company_name': company_name,
        'company_description': company_description
    }
    
    logger.info(f"Starting test with {args.iterations} iterations")
    SalesGuru(csv_file_path=csv_file).test(
        n_iterations=args.iterations, 
        openai_model_name=args.model, 
        inputs=inputs
    )
    logger.info("Test completed successfully")


def replay_with_args(args) -> None:
    """Replay the crew execution with command line arguments."""
    task_id = args.task_id
    if not task_id:
        raise ValueError("Task ID is required for replay")
    
    logger.info(f"Starting replay for task ID: {task_id}")
    SalesGuru().replay(task_id=task_id)
    logger.info("Replay completed successfully")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sales Guru - AI-powered sales automation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sales-guru                                    # Interactive mode
  sales-guru --csv leads.csv                   # Specify CSV file
  sales-guru --csv data.csv --company "Acme Corp" --description "Software company"
  sales-guru train                             # Train the model
  sales-guru test --csv test_data.csv          # Test the model
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='run',
        choices=['run', 'train', 'test', 'replay'],
        help='Command to execute (default: run)'
    )
    
    parser.add_argument(
        '--csv', '--csv-file',
        type=str,
        help=f'Path to CSV file containing leads data (default: {config.default_csv_path})'
    )
    
    parser.add_argument(
        '--company', '--company-name',
        type=str,
        help='Your company name (default: Oceaneering Mobile Robotics)'
    )
    
    parser.add_argument(
        '--description', '--company-description',
        type=str,
        help='Your company description'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Training specific arguments
    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='Number of training/test iterations (default: 1)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='OpenAI model name for testing'
    )
    
    parser.add_argument(
        '--task-id',
        type=str,
        help='Task ID for replay command'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output filename for training results'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point that handles command line arguments."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging based on environment and verbosity
    if os.getenv('ENVIRONMENT', 'development') == 'production':
        setup_production_logging()
    else:
        setup_development_logging()
    
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    try:
        if args.command == "run":
            run(
                csv_file=args.csv,
                company_name=args.company,
                company_description=args.description
            )
        elif args.command == "train":
            train_with_args(args)
        elif args.command == "test":
            test_with_args(args)
        elif args.command == "replay":
            replay_with_args(args)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
