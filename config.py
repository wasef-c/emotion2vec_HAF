import torch
import os
import logging


# Environment configuration
os.environ["FUNASR_DISABLE_PROGRESS_BAR"] = "1"
os.environ["FUNASR_QUIET"] = "1"
os.environ["FUNASR_LOG_LEVEL"] = "ERROR"
os.environ["FUNASR_DISABLE_TQDM"] = "1"

# Set up logging
logging.getLogger("funasr").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)

# Default curriculum learning epochs
DEFAULT_CURR_EPOCHS = 15


class ExperimentConfig:
    """Configuration class for emotion recognition experiments."""
    
    def __init__(self, experiment_name=None, wandb_project="emorec-adaptive-saliency"):
        self.wandb_project = wandb_project
        
        # Model configuration
        self.pretrain_dir = None
        self.use_speaker_disentanglement = True
        self.use_curriculum_learning = True
        
        # Training configuration
        self.batch_size = 38
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Class weights for handling imbalanced data
        self.class_weights = {
            "neutral": 1.0,
            "happy": 3.0,  # Higher weight for happy class
            "sad": 1.0,
            "anger": 1.0,
        }
        
        # Early stopping configuration
        self.early_stopping = {
            "patience": 10,
            "min_delta": 0.00001,  # Minimum change in monitored value to qualify as an improvement
        }
        
        # Curriculum learning configuration
        self.curriculum = {
            "start_threshold": 0.0,  # Start with easiest samples
            "end_threshold": 1.0,  # End with all samples
        }
        self.curriculum_epochs = DEFAULT_CURR_EPOCHS  # Number of epochs for curriculum learning
        
        # Dataset configuration
        self.class_names = ["neutral", "happy", "sad", "anger"]
        
        # Generate experiment name automatically if not provided
        if experiment_name is None:
            self.experiment_name = self._generate_experiment_name()
        else:
            self.experiment_name = experiment_name
    
    def _generate_experiment_name(self):
        """Generate experiment name based on configuration settings."""
        components = []
        
        # Add adaptive saliency indicator
        components.append("ADS")
        
        # Add curriculum learning flag with epochs
        if self.use_curriculum_learning:
            if self.curriculum_epochs != 20:  # Only show if different from default
                components.append(f"CURR{self.curriculum_epochs}")
            else:
                components.append("CURR")
        
        # Add speaker disentanglement flag  
        if self.use_speaker_disentanglement:
            components.append("SD")
        
        # Add class weights signature
        # Extract key weights that differ from 1.0
        weight_parts = []
        for class_name, weight in self.class_weights.items():
            if weight != 1.0:
                # Use first letter of class name and weight
                class_abbrev = class_name[0].upper()
                if weight == int(weight):
                    weight_str = str(int(weight))
                else:
                    weight_str = f"{weight:.1f}".replace(".", "p")
                weight_parts.append(f"{class_abbrev}{weight_str}")
        
        if weight_parts:
            components.extend(weight_parts)
        
        # Add learning rate if different from default
        if self.learning_rate != 1e-4:
            lr_str = f"LR{self.learning_rate:.0e}".replace("e-0", "e-").replace("e-", "m")
            components.append(lr_str)
        
        # Add batch size if different from default
        if self.batch_size != 38:
            components.append(f"BS{self.batch_size}")
        
        # Join components with underscore
        experiment_name = "_".join(components)
        
        return experiment_name
    
    def regenerate_experiment_name(self):
        """Regenerate experiment name after changing configuration."""
        self.experiment_name = self._generate_experiment_name()
        return self.experiment_name
        
    def to_dict(self):
        """Convert configuration to dictionary for logging."""
        return {
            "experiment_name": self.experiment_name,
            "wandb_project": self.wandb_project,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "class_weights": self.class_weights,
            "early_stopping": self.early_stopping,
            "curriculum": self.curriculum,
            "curriculum_epochs": self.curriculum_epochs,
            "use_speaker_disentanglement": self.use_speaker_disentanglement,
            "use_curriculum_learning": self.use_curriculum_learning,
            "pretrain_dir": self.pretrain_dir,
        }


def get_default_config():
    """Get default configuration for experiments."""
    return ExperimentConfig()


def get_curriculum_epochs():
    """Get the default curriculum epochs value."""
    return DEFAULT_CURR_EPOCHS