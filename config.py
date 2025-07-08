"""
Configuration for emotion recognition experiments
"""
import torch
import os

# Environment setup
os.environ["FUNASR_DISABLE_PROGRESS_BAR"] = "1"
os.environ["FUNASR_QUIET"] = "1"
os.environ["FUNASR_LOG_LEVEL"] = "ERROR"

class Config:
    """Simple configuration class"""
    
    def __init__(self):
        # Basic settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 20
        self.num_epochs = 50
        self.learning_rate = 3e-4
        self.weight_decay = 1e-4
        self.lr_scheduler = "cosine"  # Default to cosine (original behavior)
        
        # Class weights
        self.class_weights = {
            "neutral": 1.0,
            "happy": 2.5,
            "sad": 1.5,
            "anger": 1.5
        }
        
        # Curriculum learning
        self.use_curriculum_learning = True
        self.curriculum_epochs = 15
        self.curriculum_pacing = "exponential"
        
        # Speaker disentanglement
        self.use_speaker_disentanglement = False
        
        # Early stopping
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 0.00001
        
        # Experiment settings
        self.wandb_project = "emotion2vec_difficulty"
        self.single_test_session = 5  # Test only on session 5 for speed
        
        # Expected VAD values for difficulty calculation
        self.expected_vad = {
            0: [2.5, 2.5, 2.5],  # neutral
            1: [3.8, 3.2, 3.2],  # happy
            2: [1.8, 2.0, 2.0],  # sad 
            3: [2.0, 3.8, 3.5],  # anger
        }
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lr_scheduler": self.lr_scheduler,
            "class_weights": self.class_weights,
            "use_curriculum_learning": self.use_curriculum_learning,
            "curriculum_epochs": self.curriculum_epochs,
            "curriculum_pacing": self.curriculum_pacing,
            "use_speaker_disentanglement": self.use_speaker_disentanglement,
            "single_test_session": self.single_test_session,
        }