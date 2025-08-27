#!/usr/bin/env python3
"""
Model Preloader for Protein Structure Prediction
Handles safe model preloading with resource monitoring and graceful fallback
"""

import os
import gc
import time
import threading
from typing import Dict, Tuple, Optional, Callable
from final_workflow4 import load_model_and_tokenizer

class ModelPreloader:
    """Handles model preloading with resource monitoring and graceful fallback"""
    
    def __init__(self):
        self.models = {}  # Cache for preloaded models
        self.preload_status = {
            "completed": False,
            "success": False,
            "models_loaded": 0,
            "total_models": 3,
            "current_task": "Initializing...",
            "errors": []
        }
        self.model_paths = [
            "./models/Uniprot_Experimental_Model",      # Most common - load first
            "./models/Transmembrane_Experimental_Model", 
            "./models/Disordered_DSSP_Model_PDB"
        ]
    
    def check_system_resources(self) -> Dict[str, bool]:
        """Check if system has sufficient resources for preloading"""
        try:
            # Check available memory (rough estimate)
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                # Windows memory check
                result = subprocess.run(['wmic', 'OS', 'get', 'TotalVisibleMemorySize', '/value'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'TotalVisibleMemorySize' in line:
                            memory_kb = int(line.split('=')[1])
                            memory_gb = memory_kb / (1024 * 1024)
                            break
                else:
                    memory_gb = 8  # Default assumption
            else:
                # Linux/Mac memory check
                try:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            memory_kb = int(line.split()[1])
                            memory_gb = memory_kb / (1024 * 1024)
                            break
                except:
                    memory_gb = 8  # Default assumption
            
            # Check CPU cores
            cpu_cores = os.cpu_count() or 4
            
            # Resource requirements for preloading
            min_memory_gb = 6  # Minimum 6GB for safe preloading
            min_cpu_cores = 2  # Minimum 2 cores
            
            return {
                "sufficient_memory": memory_gb >= min_memory_gb,
                "sufficient_cpu": cpu_cores >= min_cpu_cores,
                "total_memory_gb": memory_gb,
                "cpu_cores": cpu_cores,
                "can_preload": memory_gb >= min_memory_gb and cpu_cores >= min_cpu_cores
            }
            
        except Exception as e:
            print(f"⚠️ Could not check system resources: {e}")
            # Conservative fallback - assume limited resources
            return {
                "sufficient_memory": False,
                "sufficient_cpu": True,
                "total_memory_gb": 4,
                "cpu_cores": 2,
                "can_preload": False
            }
    
    def preload_models_safe(self, progress_callback: Optional[Callable] = None) -> bool:
        """Safely preload models with resource monitoring"""
        
        # Check system resources first
        resources = self.check_system_resources()
        
        if progress_callback:
            progress_callback(0, len(self.model_paths) + 1, "Checking system resources...")
        
        print(f"💻 System Resources:")
        print(f"   Memory: {resources['total_memory_gb']:.1f} GB")
        print(f"   CPU Cores: {resources['cpu_cores']}")
        print(f"   Can Preload: {'✅ Yes' if resources['can_preload'] else '❌ No'}")
        
        if not resources['can_preload']:
            print("⚠️ Insufficient resources for preloading. Will use on-demand loading.")
            self.preload_status["completed"] = True
            self.preload_status["success"] = False
            self.preload_status["current_task"] = "Using on-demand loading (resource limited)"
            if progress_callback:
                progress_callback(len(self.model_paths) + 1, len(self.model_paths) + 1, 
                                "Ready (on-demand loading)")
            return False
        
        # Attempt to preload models
        models_loaded = 0
        
        for i, model_path in enumerate(self.model_paths):
            model_name = os.path.basename(model_path)
            
            if progress_callback:
                progress_callback(i + 1, len(self.model_paths) + 1, f"Loading {model_name}...")
            
            self.preload_status["current_task"] = f"Loading {model_name}..."
            
            try:
                print(f"🔄 Preloading: {model_name}")
                start_time = time.time()
                
                # Load model with error handling
                model, tokenizer = load_model_and_tokenizer(model_path)
                
                # Store in cache
                self.models[model_path] = (model, tokenizer)
                models_loaded += 1
                
                load_time = time.time() - start_time
                print(f"✅ Preloaded {model_name} in {load_time:.2f}s")
                
                # Force garbage collection to free memory
                gc.collect()
                
            except Exception as e:
                error_msg = f"Failed to preload {model_name}: {e}"
                print(f"⚠️ {error_msg}")
                self.preload_status["errors"].append(error_msg)
                
                # If we fail to load any model, stop preloading
                if models_loaded == 0:
                    print("❌ Failed to load any models. Falling back to on-demand loading.")
                    self.preload_status["completed"] = True
                    self.preload_status["success"] = False
                    self.preload_status["current_task"] = "Using on-demand loading (preload failed)"
                    if progress_callback:
                        progress_callback(len(self.model_paths) + 1, len(self.model_paths) + 1,
                                        "Ready (on-demand loading)")
                    return False
        
        # Update status
        self.preload_status["models_loaded"] = models_loaded
        self.preload_status["completed"] = True
        self.preload_status["success"] = models_loaded > 0
        
        if models_loaded == len(self.model_paths):
            self.preload_status["current_task"] = f"🎉 All {models_loaded} models ready!"
            print(f"🎉 Successfully preloaded all {models_loaded} models!")
        else:
            self.preload_status["current_task"] = f"⚠️ {models_loaded}/{len(self.model_paths)} models loaded"
            print(f"⚠️ Partially loaded: {models_loaded}/{len(self.model_paths)} models")
        
        if progress_callback:
            progress_callback(len(self.model_paths) + 1, len(self.model_paths) + 1,
                            self.preload_status["current_task"])
        
        return models_loaded > 0
    
    def get_model(self, model_path: str) -> Optional[Tuple]:
        """Get preloaded model or None if not available"""
        return self.models.get(model_path, None)
    
    def is_model_preloaded(self, model_path: str) -> bool:
        """Check if a specific model is preloaded"""
        return model_path in self.models
    
    def get_preload_status(self) -> Dict:
        """Get current preloading status"""
        return self.preload_status.copy()
    
    def clear_cache(self):
        """Clear all preloaded models to free memory"""
        print("🗑️ Clearing preloaded models...")
        self.models.clear()
        gc.collect()
        print("✅ Model cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get information about preloaded models"""
        return {
            "preloaded_models": list(self.models.keys()),
            "models_count": len(self.models),
            "preload_completed": self.preload_status["completed"],
            "preload_successful": self.preload_status["success"]
        }

# Global preloader instance
_global_preloader = None

def get_global_preloader() -> ModelPreloader:
    """Get the global model preloader instance"""
    global _global_preloader
    if _global_preloader is None:
        _global_preloader = ModelPreloader()
    return _global_preloader

def preload_models_threaded(progress_callback: Optional[Callable] = None) -> threading.Thread:
    """Start model preloading in a separate thread"""
    preloader = get_global_preloader()
    
    def preload_worker():
        preloader.preload_models_safe(progress_callback)
    
    thread = threading.Thread(target=preload_worker, daemon=True)
    thread.start()
    return thread

def get_preloaded_model(model_path: str) -> Optional[Tuple]:
    """Get preloaded model or None if not available"""
    preloader = get_global_preloader()
    return preloader.get_model(model_path)

def is_preloading_complete() -> bool:
    """Check if preloading is complete"""
    preloader = get_global_preloader()
    return preloader.preload_status["completed"]

def get_preloading_status() -> Dict:
    """Get current preloading status"""
    preloader = get_global_preloader()
    return preloader.get_preload_status()
