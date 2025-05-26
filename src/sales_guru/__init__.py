"""Sales Guru - AI-powered sales automation and lead management system.

This package provides a crew of specialized AI agents for automating 
sales processes including lead qualification, prospect research,
email outreach, and sales call preparation.
"""

from sales_guru.crew import SalesGuru
from sales_guru.config import config
from sales_guru.error_handling import (
    SalesGuruError, APIError, RateLimitError, NetworkError, ConfigurationError
)

__version__ = "1.0.0"
__author__ = "SalesGuru Team"
__all__ = [
    'SalesGuru', 
    'config',
    'SalesGuruError', 
    'APIError', 
    'RateLimitError', 
    'NetworkError', 
    'ConfigurationError'
]

# This is the main entry point for the sales_guru package.
# You can import the SalesGuru class from here to use it in your code.
# Example:
# from sales_guru import SalesGuru
# crew = SalesGuru()
# crew.kickoff()
