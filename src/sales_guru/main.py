#!/usr/bin/env python
import sys
import warnings
#import argparse  # removed

from sales_guru.crew import SalesGuru

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.

def run():
    """
    Run the crew interactively with task completion guarantees.
    """
    company_name = input("Enter the company name: ")
    company_description = input("Enter the company description: ")
    inputs = {
        'company_name': company_name,
        'company_description': company_description
    }
    try:
        SalesGuru().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew interactively with task completion guarantees.
    """
    company_name = input("Enter the company name: ")
    company_description = input("Enter the company description: ")
    try:
        iterations = int(input("Enter number of training iterations: "))
    except ValueError:
        raise Exception("Invalid number for training iterations")
    filename = input("Enter filename to save training results: ")
    inputs = {
        'company_name': company_name,
        'company_description': company_description
    }
    try:
        SalesGuru().train(n_iterations=iterations, filename=filename, inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution interactively with task completion guarantees.
    """
    task_id = input("Enter task ID to replay: ")
    try:
        SalesGuru().replay(task_id=task_id)
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution interactively with task completion guarantees and returns the results.
    """
    company_name = input("Enter the company name: ")
    company_description = input("Enter the company description: ")
    try:
        iterations = int(input("Enter number of test iterations: "))
    except ValueError:
        raise Exception("Invalid number for test iterations")
    model_name = input("Enter OpenAI model name to use for testing: ")
    inputs = {
        'company_name': company_name,
        'company_description': company_description
    }
    try:
        SalesGuru().test(n_iterations=iterations, openai_model_name=model_name, inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    # Check if a command was passed (train, replay, test) otherwise run interactively
    if len(sys.argv) > 1 and sys.argv[1] in ['train', 'replay', 'test']:
        command = sys.argv.pop(1)  # Remove the command from argv
        if command == 'train':
            train()
        elif command == 'replay':
            replay()
        elif command == 'test':
            test()
    else:
        run()
