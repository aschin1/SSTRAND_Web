#!/usr/bin/env python3
"""
Test script to validate SSTRAND setup
This script checks if all components are working correctly
"""

import sys
import os
import json
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing Python imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("💻 CUDA not available (using CPU)")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        from transformers import AutoModel, AutoTokenizer
        print("✅ Transformers library")
    except ImportError:
        print("❌ Transformers not found")
        return False
    
    try:
        from Bio import SeqIO
        print("✅ BioPython")
    except ImportError:
        print("❌ BioPython not found")
        return False
    
    try:
        import platform_utils
        print("✅ platform_utils")
    except ImportError:
        print("❌ platform_utils not found")
        return False
    
    try:
        from model.accession_retrieval4 import get_accession_and_sequence
        print("✅ model.accession_retrieval4")
    except ImportError:
        print("❌ model.accession_retrieval4 not found")
        return False
    
    try:
        from classifiers.main_classifier import classify_sequence
        print("✅ classifiers.main_classifier")
    except ImportError:
        print("❌ classifiers.main_classifier not found")
        return False
    
    try:
        import final_workflow4
        print("✅ final_workflow4")
    except ImportError:
        print("❌ final_workflow4 not found")
        return False
    
    return True

def test_models():
    """Test if models are properly set up"""
    print("\n🤖 Testing model setup...")
    
    try:
        from platform_utils import get_model_path, get_available_models
        
        available_models = get_available_models()
        print(f"📦 Available models: {available_models}")
        
        required_models = ['membrane', 'disordered', 'structured']
        missing_models = set(required_models) - set(available_models)
        
        if missing_models:
            print(f"⚠️ Missing models: {', '.join(missing_models)}")
            for model_type in missing_models:
                model_dir = Path('models') / model_type
                if model_dir.exists():
                    files = list(model_dir.glob('*'))
                    print(f"   {model_type} directory contains: {[f.name for f in files]}")
                else:
                    print(f"   {model_type} directory not found")
            return False
        
        print("✅ All required models found")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_workflow():
    """Test the complete workflow with a simple sequence"""
    print("\n🧬 Testing workflow...")
    
    try:
        from final_workflow4 import run_final_workflow
        
        # Test sequence (human insulin)
        test_sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
        
        print(f"🔬 Running workflow with test sequence ({len(test_sequence)} residues)...")
        
        # Run the workflow
        result = run_final_workflow(test_sequence)
        
        if result:
            print("✅ Workflow completed successfully")
            print(f"   Classification: {result.get('classification', 'Unknown')}")
            print(f"   Model used: {result.get('model_used', 'Unknown')}")
            print(f"   Structure length: {len(result.get('combined_structure', ''))}")
            print(f"   Processing time: {result.get('time', 'Unknown')}")
            return True
        else:
            print("❌ Workflow returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_processor():
    """Test the JSON processor if available"""
    print("\n📄 Testing JSON processor...")
    
    if not os.path.exists('workflow_json_processor.py'):
        print("⚠️ workflow_json_processor.py not found - will use direct workflow calls")
        return True
    
    try:
        from workflow_json_processor import process_sequence_request, format_for_web_display
        
        # Test input
        input_data = {
            'sequence': 'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN',
            'job_id': 'test_001'
        }
        
        print("🔄 Processing test sequence through JSON interface...")
        
        # Process request
        result = process_sequence_request(input_data)
        
        if result.get('success'):
            # Format for web display
            formatted_result = format_for_web_display(result)
            
            print("✅ JSON processor working correctly")
            print(f"   Job ID: {formatted_result.get('job_id')}")
            print(f"   Classification: {formatted_result.get('classification')}")
            print(f"   Aligned blocks: {len(formatted_result.get('aligned_display', []))}")
            return True
        else:
            print(f"❌ JSON processor failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ JSON processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_connectivity():
    """Test network connectivity for external services"""
    print("\n🌐 Testing network connectivity...")
    
    try:
        import requests
        
        test_urls = [
            ('UniProt', 'https://rest.uniprot.org/'),
            ('RCSB PDB', 'https://files.rcsb.org/'),
            ('NCBI BLAST', 'https://blast.ncbi.nlm.nih.gov/')
        ]
        
        for name, url in test_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"✅ {name} accessible")
                else:
                    print(f"⚠️ {name} returned status {response.status_code}")
            except:
                print(f"❌ Cannot reach {name}")
        
        return True
        
    except ImportError:
        print("⚠️ requests library not available - skipping network tests")
        return True

def create_test_files():
    """Create test files for validation"""
    print("\n📁 Creating test files...")
    
    # Test FASTA file
    test_fasta = """>sp|P01308|INS_HUMAN Insulin OS=Homo sapiens OX=9606 GN=INS PE=1 SV=1
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"""
    
    with open('test_insulin.fasta', 'w') as f:
        f.write(test_fasta)
    
    # Test JSON input
    test_json = {
        "sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
        "job_id": "test_job_001",
        "timestamp": "2025-08-22T12:00:00Z"
    }
    
    with open('test_input.json', 'w') as f:
        json.dump(test_json, f, indent=2)
    
    print("✅ Test files created:")
    print("   - test_insulin.fasta")
    print("   - test_input.json")

def check_directory_structure():
    """Check if directory structure is correct"""
    print("\n📂 Checking directory structure...")
    
    required_dirs = [
        'model',
        'classifiers',
        'models',
        'models/membrane',
        'models/disordered',
        'models/structured',
        'pdb_files',
        'uploads',
        'public',
        'views'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"⚠️ Missing directories: {missing_dirs}")
        # Create missing directories
        for dir_path in missing_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        print("✅ Created missing directories")
    else:
        print("✅ Directory structure is correct")
    
    return True

def main():
    """Main test function"""
    print("🧪 SSTRAND Setup Validation")
    print("=" * 50)
    
    # Track test results
    test_results = []
    
    # Run all tests
    test_results.append(("Directory Structure", check_directory_structure()))
    test_results.append(("Python Imports", test_imports()))
    test_results.append(("Model Setup", test_models()))
    test_results.append(("Network Connectivity", test_network_connectivity()))
    test_results.append(("Workflow", test_workflow()))
    test_results.append(("JSON Processor", test_json_processor()))
    
    # Create test files
    create_test_files()
    
    # Summary
    print("\n📊 Test Summary")
    print("-" * 30)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your SSTRAND setup is ready.")
        print("\nNext steps:")
        print("1. Start the server: node server.js")
        print("2. Open browser: http://localhost:5050")
        print("3. Test with test_insulin.fasta")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the issues above.")
        print("\nCommon fixes:")
        print("- Install missing Python packages: pip install -r requirements.txt")
        print("- Ensure model files are in the correct directories")
        print("- Check network connectivity")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)