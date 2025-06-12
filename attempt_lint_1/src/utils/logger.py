import logging
import os
from datetime import datetime

class Logger:
    """
    Centralized logging utility for training progress, validation results, and system messages.
    """
    def __init__(self, log_dir, log_file, level="INFO"):
        self.log_dir = log_dir
        self.log_file = log_file
        self.level = level

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.level.upper()))
        self.logger.propagate = False # Prevent duplicate logs from root logger

        # Clear existing handlers to avoid duplicates on re-init
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)

        # File handler
        file_path = os.path.join(self.log_dir, self.log_file)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)