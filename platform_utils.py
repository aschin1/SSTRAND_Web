import os
from pathlib import Path

class PlatformManager:
    def __init__(self):
        self.app_root = Path(__file__).parent
        
    def get_model_root(self):
        return self.app_root / "models"
    
    def get_pdb_root(self):
        return self.app_root / "pdb_files"
    
    def get_blast_root(self):
        return self.app_root / "blast_db"
    
    def get_uploads_root(self):
        return self.app_root / "uploads"
    
    def get_results_root(self):
        return self.app_root / "results"

# Global instance
platform_manager = PlatformManager()

def get_model_path(model_type):
    """
    Get path to specific model directory
    
    Args:
        model_type: 'membrane', 'disordered', or 'structured'
        
    Returns:
        str: Path to model directory
    """
    model_root = platform_manager.get_model_root()
    model_path = model_root / model_type
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    return str(model_path)

def ensure_directories():
    """Ensure all required directories exist"""
    dirs_to_create = [
        platform_manager.get_model_root(),
        platform_manager.get_pdb_root(),
        platform_manager.get_blast_root(),
        platform_manager.get_uploads_root(),
        platform_manager.get_results_root()
    ]
    
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ All required directories ensured")

def get_available_models():
    """Get list of available model types"""
    model_root = platform_manager.get_model_root()
    available_models = []
    
    for model_type in ['membrane', 'disordered', 'structured']:
        model_path = model_root / model_type
        if model_path.exists():
            # Check for required model files
            config_file = model_path / "config.json"
            model_file = model_path / "pytorch_model.bin"
            safetensors_file = model_path / "model.safetensors"
            
            if config_file.exists() and (model_file.exists() or safetensors_file.exists()):
                available_models.append(model_type)
    
    return available_models

def validate_model_setup():
    """Validate that models are properly set up"""
    available_models = get_available_models()
    required_models = ['membrane', 'disordered', 'structured']
    
    missing_models = set(required_models) - set(available_models)
    
    if missing_models:
        print(f"⚠️ Missing models: {', '.join(missing_models)}")
        return False
    
    print("✅ All required models are available")
    return True