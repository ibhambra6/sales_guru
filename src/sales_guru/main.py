#!/usr/bin/env python
import sys
import warnings

from sales_guru.crew import SalesGuru

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'company_name': 'Oceaneering Mobile Robotics',
        'company_description': 'A division of Oceaneering International that designs, manufactures, and maintains innovative mobile robotics solutions for material handling and logistics challenges. With over 30 years of experience, OMR has deployed more than 1,700 robots globally across various industries. Their product portfolio includes autonomous mobile robots (AMRs) like the UniMover series and MaxMover forklifts, featuring natural feature navigation, safety lidar, and high-performance battery systems. OMR robots can operate in either AMR mode with obstacle avoidance or AGV mode for increased speed, and can work together as mixed fleets to provide comprehensive solutions.'
    }
    
    try:
        SalesGuru().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'company_name': 'Oceaneering Mobile Robotics',
        'company_description': 'A division of Oceaneering International that designs, manufactures, and maintains innovative mobile robotics solutions for material handling and logistics challenges. With over 30 years of experience, OMR has deployed more than 1,700 robots globally across various industries. Their product portfolio includes autonomous mobile robots (AMRs) like the UniMover series and MaxMover forklifts, featuring natural feature navigation, safety lidar, and high-performance battery systems. OMR robots can operate in either AMR mode with obstacle avoidance or AGV mode for increased speed, and can work together as mixed fleets to provide comprehensive solutions.'
    }
    try:
        SalesGuru().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        SalesGuru().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'company_name': 'Oceaneering Mobile Robotics',
        'company_description': 'A division of Oceaneering International that designs, manufactures, and maintains innovative mobile robotics solutions for material handling and logistics challenges. With over 30 years of experience, OMR has deployed more than 1,700 robots globally across various industries. Their product portfolio includes autonomous mobile robots (AMRs) like the UniMover series and MaxMover forklifts, featuring natural feature navigation, safety lidar, and high-performance battery systems. OMR robots can operate in either AMR mode with obstacle avoidance or AGV mode for increased speed, and can work together as mixed fleets to provide comprehensive solutions.'
    }
    try:
        SalesGuru().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
