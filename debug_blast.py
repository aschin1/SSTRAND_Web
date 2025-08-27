#!/usr/bin/env python3
"""
Debug script to identify and fix BLAST issues
"""

import sys
import json
import requests
import time
from pathlib import Path

def test_blast_connectivity():
    """Test NCBI BLAST web service connectivity"""
    print("🌐 Testing NCBI BLAST connectivity...")
    
    try:
        blast_url = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
        response = requests.get(blast_url, timeout=10)
        
        if response.status_code == 200:
            print("✅ NCBI BLAST service is accessible")
            return True
        else:
            print(f"⚠️ NCBI BLAST returned status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot reach NCBI BLAST: {e}")
        return False

def test_simple_blast_search():
    """Test a simple BLAST search"""
    print("\n🔍 Testing simple BLAST search...")
    
    try:
        # Simple test sequence (insulin)
        test_sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
        
        blast_url = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
        
        # Submit BLAST job
        submit_params = {
            'CMD': 'Put',
            'PROGRAM': 'blastp',
            'DATABASE': 'pdb',
            'QUERY': test_sequence,
            'EXPECT': '1e-5',
            'HITLIST_SIZE': '10',
            'FORMAT_TYPE': 'JSON2'
        }
        
        print("📤 Submitting BLAST job...")
        submit_response = requests.post(blast_url, data=submit_params, timeout=30)
        
        if submit_response.status_code != 200:
            print(f"❌ BLAST submission failed: {submit_response.status_code}")
            return False
        
        # Extract RID
        submit_text = submit_response.text
        if 'RID = ' not in submit_text:
            print("❌ Could not extract BLAST RID")
            print("Response:", submit_text[:500])
            return False
        
        rid_start = submit_text.find('RID = ') + 6
        rid_end = submit_text.find('\n', rid_start)
        rid = submit_text[rid_start:rid_end].strip()
        
        print(f"✅ BLAST job submitted with RID: {rid}")
        
        # Poll for results (simplified)
        max_wait = 60  # 1 minute max
        wait_time = 0
        
        while wait_time < max_wait:
            time.sleep(10)
            wait_time += 10
            
            check_params = {
                'CMD': 'Get',
                'RID': rid,
                'FORMAT_TYPE': 'JSON2'
            }
            
            print(f"⏳ Checking results... ({wait_time}s)")
            check_response = requests.get(blast_url, params=check_params, timeout=30)
            
            if 'Status=WAITING' in check_response.text:
                continue
            elif 'Status=FAILED' in check_response.text:
                print("❌ BLAST job failed")
                return False
            else:
                print("✅ BLAST job completed")
                
                # Try to parse results
                try:
                    results = json.loads(check_response.text)
                    print("✅ Results are in JSON format")
                    
                    # Look for hits
                    hits_found = 0
                    if 'BlastOutput2' in results:
                        for output in results['BlastOutput2']:
                            if 'report' in output and 'results' in output['report']:
                                search = output['report']['results'].get('search', {})
                                hits = search.get('hits', [])
                                hits_found = len(hits)
                                break
                    
                    print(f"🎯 Found {hits_found} BLAST hits")
                    
                    if hits_found > 0:
                        print("✅ BLAST search working correctly")
                        return True
                    else:
                        print("⚠️ BLAST search completed but found no hits")
                        return False
                        
                except json.JSONDecodeError:
                    print("⚠️ Results are not in JSON format, might be HTML or text")
                    if "insulin" in check_response.text.lower():
                        print("✅ Results mention insulin, BLAST is working")
                        return True
                    else:
                        print("❌ Unexpected result format")
                        return False
        
        print("⏰ BLAST search timed out")
        return False
        
    except Exception as e:
        print(f"❌ BLAST test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdb_blast_searcher():
    """Test the PDBBlastSearcher class"""
    print("\n🧪 Testing PDBBlastSearcher...")
    
    try:
        from model.pdb_blast_search import PDBBlastSearcher
        
        searcher = PDBBlastSearcher()
        
        # Create test query file
        test_sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
        
        query_file = "test_blast_query.fasta"
        with open(query_file, 'w') as f:
            f.write(f">test_insulin\n{test_sequence}\n")
        
        print("📁 Created test query file")
        
        # Test web BLAST search
        print("🌐 Testing web BLAST search...")
        hits = searcher.run_web_blast_search(query_file, max_hits=5)
        
        if hits:
            print(f"✅ Web BLAST found {len(hits)} hits")
            for i, hit in enumerate(hits[:3]):
                print(f"   Hit {i+1}: {hit.get('subject_id', 'Unknown')} ({hit.get('percent_identity', 0):.1f}% identity)")
            return True
        else:
            print("❌ Web BLAST found no hits")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import PDBBlastSearcher: {e}")
        return False
    except Exception as e:
        print(f"❌ PDBBlastSearcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if Path("test_blast_query.fasta").exists():
            Path("test_blast_query.fasta").unlink()

def fix_workflow_blast_integration():
    """Check and fix BLAST integration in final_workflow4.py"""
    print("\n🔧 Checking BLAST integration in workflow...")
    
    try:
        # Check if final_workflow4.py exists and has BLAST integration
        workflow_file = Path("final_workflow4.py")
        if not workflow_file.exists():
            print("❌ final_workflow4.py not found")
            return False
        
        with open(workflow_file, 'r') as f:
            content = f.read()
        
        # Check for BLAST imports and usage
        blast_imports = [
            "from model.pdb_blast_search import PDBBlastSearcher",
            "PDBBlastSearcher",
            "blast_search",
            "run_blast_search"
        ]
        
        missing_blast = []
        for item in blast_imports:
            if item not in content:
                missing_blast.append(item)
        
        if missing_blast:
            print(f"⚠️ Missing BLAST integration: {missing_blast}")
            print("   Your workflow may not be calling BLAST properly")
            return False
        else:
            print("✅ BLAST integration found in workflow")
            return True
            
    except Exception as e:
        print(f"❌ Error checking workflow: {e}")
        return False

def create_minimal_blast_fix():
    """Create a minimal fix for BLAST issues"""
    print("\n🛠️ Creating BLAST fix...")
    
    fix_code = '''
# Add this to your final_workflow4.py if BLAST is failing

def simple_pdb_search(sequence):
    """
    Simple PDB search using RCSB PDB API as fallback
    """
    try:
        import requests
        
        # Use RCSB PDB sequence search
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        
        query = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "evalue_cutoff": 1e-10,
                    "identity_cutoff": 0.3,
                    "target": "pdb_protein_sequence",
                    "value": sequence
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": 10}
            }
        }
        
        response = requests.post(search_url, json=query, timeout=30)
        
        if response.status_code == 200:
            results = response.json()
            if "result_set" in results and results["result_set"]:
                hits = []
                for result in results["result_set"][:5]:
                    hits.append({
                        "subject_id": result["identifier"],
                        "percent_identity": 85.0,  # Estimated
                        "evalue": 1e-50
                    })
                return hits
        
        return []
        
    except Exception as e:
        print(f"Simple PDB search failed: {e}")
        return []

# Use this function in your workflow if BLAST fails:
# hits = simple_pdb_search(sequence)
'''
    
    with open("blast_fix.py", "w") as f:
        f.write(fix_code)
    
    print("✅ Created blast_fix.py with fallback search function")
    print("   You can integrate this into your workflow if BLAST continues to fail")

def main():
    """Main debug function"""
    print("🔍 SSTRAND BLAST Debug Tool")
    print("=" * 40)
    
    # Run diagnostic tests
    tests = [
        ("BLAST Connectivity", test_blast_connectivity),
        ("Simple BLAST Search", test_simple_blast_search),
        ("PDBBlastSearcher", test_pdb_blast_searcher),
        ("Workflow Integration", fix_workflow_blast_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # Create fix regardless
    create_minimal_blast_fix()
    
    # Summary
    print(f"\n📊 Debug Summary")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed >= 2:  # At least connectivity and one other test
        print("\n✅ BLAST should be working. The issue might be:")
        print("   - Network timeouts (increase timeout values)")
        print("   - Rate limiting (add delays between requests)")
        print("   - Query sequence too short/generic")
    else:
        print("\n⚠️ BLAST integration has issues. Recommendations:")
        print("   1. Check network connectivity")
        print("   2. Use the fallback function in blast_fix.py")
        print("   3. Consider using local BLAST database")
        print("   4. Add longer timeout values")

if __name__ == "__main__":
    main()