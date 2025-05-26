"""Configuration management for Sales Guru."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Centralized configuration management."""
    
    def __init__(self):
        self.config_dir = Path(__file__).parent
        self._agents_config = None
        self._tasks_config = None
        self._llm_config = None
        
    @property
    def agents_config(self) -> Dict[str, Any]:
        """Load and cache agents configuration."""
        if self._agents_config is None:
            self._agents_config = self._load_yaml_config('agents.yaml')
        return self._agents_config
    
    @property
    def tasks_config(self) -> Dict[str, Any]:
        """Load and cache tasks configuration."""
        if self._tasks_config is None:
            self._tasks_config = self._load_yaml_config('tasks.yaml')
        return self._tasks_config
    
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Load and cache LLM configuration."""
        if self._llm_config is None:
            self._llm_config = self._load_yaml_config('llm_config.yaml')
        return self._llm_config
    
    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {filename}: {e}")
    
    # API Keys
    @property
    def serper_api_key(self) -> Optional[str]:
        """Get Serper API key from environment."""
        return os.getenv('SERPER_API_KEY')
    
    @property
    def google_api_key(self) -> Optional[str]:
        """Get Google API key from environment."""
        return os.getenv('GOOGLE_API_KEY')
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment."""
        return os.getenv('OPENAI_API_KEY')
    
    # Default paths
    @property
    def default_csv_path(self) -> str:
        """Get default CSV file path."""
        # Check for environment variable first, then fall back to default
        return os.getenv('SALES_GURU_CSV_PATH', 'knowledge/leads.csv')
    
    # LLM Configuration
    @property
    def primary_model(self) -> str:
        """Get primary LLM model."""
        return self.llm_config.get('primary_model', 'gemini/gemini-2.5-flash-preview-05-20')
    
    @property
    def fallback_models(self) -> list:
        """Get fallback LLM models."""
        return self.llm_config.get('fallback_models', ['gemini/gemini-1.5-flash'])
    
    @property
    def lead_qualification_model(self) -> str:
        """Get lead qualification specific model."""
        return self.llm_config.get('lead_qualification_model', 'gemini/gemini-2.5-pro-preview-05-06')
    
    @property
    def openai_primary_model(self) -> str:
        """Get OpenAI primary model."""
        return self.llm_config.get('openai_primary_model', 'gpt-4.1')
    
    @property
    def openai_fallback_models(self) -> list:
        """Get OpenAI fallback models."""
        return self.llm_config.get('openai_fallback_models', ['gpt-4o', 'gpt-4o-mini'])
    
    def validate_api_keys(self) -> None:
        """Validate that required API keys are present."""
        missing_keys = []
        
        if not self.serper_api_key:
            missing_keys.append('SERPER_API_KEY')
        if not self.google_api_key:
            missing_keys.append('GOOGLE_API_KEY')
        if not self.openai_api_key:
            missing_keys.append('OPENAI_API_KEY')
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")


# Global configuration instance
config = Config() 