#!/usr/bin/env python3
"""
JSON wrapper for final_workflow4.py
This script captures all output and returns only valid JSON to stdout.
"""

import sys
import json
import io
import contextlib
from datetime import datetime

# Redirect all print statements to stderr instead of stdout
class StderrLogger:
    def write(self, message):
        sys.stderr.write(message)
    def flush(self):
        sys.stderr.flush()

# Set up logging to stderr
sys.stdout = StderrLogger()

# Now import the workflow after redirecting stdout
from final_workflow4 import run_final_workflow, clean_sequence

def format_result_for_web(result, job_id, sequence):
    """Format the workflow result for web display"""
    
    # Clean the sequence
    clean_seq = clean_sequence(sequence)
    
    # Extract disorder regions from the disorder result
    disorder_regions = []
    if result.get('disorder') and hasattr(result['disorder'], 'disordered_regions'):
        disorder_regions = result['disorder'].disordered_regions
    elif result.get('disorder') and isinstance(result['disorder'], dict):
        disorder_regions = result['disorder'].get('disordered_regions', [])
    
    # Format disorder fraction
    disorder_fraction = 0.0
    if result.get('disorder'):
        if hasattr(result['disorder'], 'disorder_fraction'):
            disorder_fraction = result['disorder'].disorder_fraction
        elif isinstance(result['disorder'], dict):
            disorder_fraction = result['disorder'].get('disorder_fraction', 0.0)
    
    # Format transmembrane regions
    tm_regions_formatted = []
    if result.get('tm_regions'):
        if isinstance(result['tm_regions'], list):
            for region in result['tm_regions']:
                if isinstance(region, tuple) and len(region) == 2:
                    tm_regions_formatted.append(f"{region[0]}-{region[1]}")
                else:
                    tm_regions_formatted.append(str(region))
    
    # Get BLAST hit info
    top_hit = result.get('top hit', {})
    if isinstance(top_hit, dict):
        blast_id = top_hit.get('subject_id', 'Unknown')
        blast_identity = top_hit.get('percent_identity', 0.0)
        blast_evalue = top_hit.get('evalue', 1.0)
    else:
        blast_id = 'Unknown'
        blast_identity = 0.0
        blast_evalue = "N/A"
    
    # Calculate coverage (percentage of query sequence covered by DSSP data)
    # Coverage should not exceed 100%
    dssp_length = len(result.get('dssp', ''))
    seq_length = len(clean_seq)
    if seq_length > 0:
        # Coverage is the min of 100% or the actual coverage
        coverage = min(100.0, (dssp_length / seq_length * 100))
    else:
        coverage = 0.0

    if isinstance(blast_evalue, (int, float)):
         evalue_str = f"{blast_evalue:.2e}"
    else:
         evalue_str = str(blast_evalue)
    
    # Create metadata text
    metadata_text = f"""PROTEIN CLASSIFICATION: {result.get('classification', 'unknown')}
DISORDER FRACTION: {disorder_fraction:.3f}
DISORDERED REGIONS: {disorder_regions if disorder_regions else 'None detected'}
MEMBRANE: {result.get('membrane', 'non-membrane')}
TRANSMEMBRANE REGIONS: {', '.join(tm_regions_formatted) if tm_regions_formatted else 'None detected'}
TOP BLAST HIT: {blast_id} | PERCENT IDENTITY: {blast_identity:.2f}% | E-VALUE: {evalue_str} | COVERAGE: {coverage:.2f}%
ACCESSION: {result.get('accession', 'Unknown')}
MODEL USED: {result.get('model_used', 'Unknown')}
TIME: {result.get('time', 'Unknown')}"""
    
    # Create aligned display for web with enhanced structure coloring
    aligned_display = []
    structure = result.get('combined structure', '')
    mask = result.get('structure mask', [])
    
    block_size = 60
    for i in range(0, len(clean_seq), block_size):
        seq_chunk = clean_seq[i:i + block_size]
        str_chunk = structure[i:i + block_size] if i < len(structure) else ''
        mask_chunk = mask[i:i + block_size] if i < len(mask) else []
        
        # Create colored versions for display
        sequence_colored = []
        structure_colored = []
        
        for j in range(len(seq_chunk)):
            seq_char = seq_chunk[j] if j < len(seq_chunk) else ''
            str_char = str_chunk[j] if j < len(str_chunk) else ''
            mask_char = mask_chunk[j] if j < len(mask_chunk) else 'M'
            
            # Amino acids get background highlighting if they have DSSP data
            seq_class = 'dssp-background' if mask_char == 'D' else ''
            sequence_colored.append({
                'char': seq_char,
                'class': seq_class
            })
            
            # Structure keeps its standard coloring based on structure type only
            str_type = ''
            if str_char == 'H':
                str_type = 'structure-helix'
            elif str_char in ['E', 'B']:
                str_type = 'structure-sheet'
            else:
                str_type = 'structure-coil'
            
            structure_colored.append({
                'char': str_char,
                'class': str_type
            })
        
        aligned_display.append({
            'position': i + 1,
            'sequence': seq_chunk,
            'structure': str_chunk,
            'mask': ''.join(mask_chunk) if mask_chunk else '',
            'sequence_colored': sequence_colored,
            'structure_colored': structure_colored
        })
        
    for i in range(0, len(clean_seq), block_size):
        seq_chunk = clean_seq[i:i + block_size]
        str_chunk = structure[i:i + block_size] if i < len(structure) else ''
        mask_chunk = mask[i:i + block_size] if i < len(mask) else []
        
        aligned_display.append({
            'position': i + 1,
            'sequence': seq_chunk,
            'structure': str_chunk,
            'mask': ''.join(mask_chunk) if mask_chunk else ''
        })
    
    # Build the final result
    return {
        'success': True,
        'job_id': job_id,
        'sequence': clean_seq,
        'structure': structure,
        'combined structure': structure,
        'structure_mask': mask,
        'classification': result.get('classification', 'unknown'),
        'disorder': {
            'disorder_fraction': str(disorder_fraction),
            'regions': str(disorder_regions) if disorder_regions else 'None detected'
        },
        'membrane': result.get('membrane', 'non-membrane'),
        'tm_regions': tm_regions_formatted,
        'top hit': top_hit,
        'identity': blast_identity,
        'e_value': blast_evalue,
        'dssp': result.get('dssp', ''),
        'accession': result.get('accession', 'Unknown'),
        'model_used': result.get('model_used', 'Unknown'),
        'time': result.get('time', 'Unknown'),
        '%helix': result.get('%helix', 0.0),
        '%sheet': result.get('%sheet', 0.0),
        '%coil': result.get('%coil', 0.0),
        'metadata_text': metadata_text,
        'aligned_display': aligned_display,
        'coverage': coverage,
        'timestamp': datetime.now().isoformat()
    }

def main():
    # Reset stdout for final JSON output
    sys.stdout = sys.__stdout__
    
    try:
        # Parse input arguments
        if len(sys.argv) < 2:
            result = {
                'success': False,
                'error': 'No input provided. Expected JSON string as argument.'
            }
            print(json.dumps(result))
            sys.exit(1)
        
        # Parse the JSON input
        input_json = sys.argv[1]
        input_data = json.loads(input_json)
        
        sequence = input_data.get('sequence', '')
        job_id = input_data.get('job_id', 'unknown')
        
        if not sequence:
            result = {
                'success': False,
                'error': 'No sequence provided in input',
                'job_id': job_id
            }
            print(json.dumps(result))
            sys.exit(1)
        
        # Redirect stdout to capture all prints from the workflow
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            # Run the actual workflow
            workflow_result = run_final_workflow(sequence)
        
        # Log the captured output to stderr for debugging
        sys.stderr.write("\n=== Captured workflow output ===\n")
        sys.stderr.write(captured_output.getvalue())
        sys.stderr.write("\n=== End captured output ===\n")
        
        if workflow_result:
            # Format the result for web display
            formatted_result = format_result_for_web(workflow_result, job_id, sequence)
            # Output ONLY the JSON to stdout
            print(json.dumps(formatted_result))
        else:
            result = {
                'success': False,
                'error': 'Workflow failed to produce results',
                'job_id': job_id
            }
            print(json.dumps(result))
            
    except json.JSONDecodeError as e:
        result = {
            'success': False,
            'error': f'Invalid JSON input: {str(e)}',
            'job_id': 'unknown'
        }
        print(json.dumps(result))
    except Exception as e:
        result = {
            'success': False,
            'error': f'Workflow error: {str(e)}',
            'job_id': input_data.get('job_id', 'unknown') if 'input_data' in locals() else 'unknown'
        }
        print(json.dumps(result))

if __name__ == '__main__':
    main()