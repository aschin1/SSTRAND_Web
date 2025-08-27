import os
import torch
import pandas as pd
import time
import requests
from transformers import AutoModelForTokenClassification, AutoTokenizer
from Bio import SeqIO
from pathlib import Path
from platform_utils import platform_manager, get_model_path
from model.accession_retrieval4 import get_accession_and_sequence, get_pdb_accession_by_sequence
from model.get_alphafold_data4 import fetch_alphafold_pdb, save_pdb_file
from model.get_pdb_data4 import get_pdb_ids_from_uniprot, fetch_pdb_structure, save_pdb_file
from model.get_secondary_struct_dssp import get_dssp_secondary_structure, smooth_secondary_structure
from model.pdb_blast_search import PDBBlastSearcher
from classifiers.main_classifier import classify_sequence
#from model.comparison_outputs import align_sequences, get_aligned_secondary_structures

"""
This script integrates various components to predict protein secondary structure
using a sliding window approach. It retrieves UniProt accessions, 
fetches AlphaFold PDB data, uses a custom classifier to determine the
type of protein (membrane, disordered, or structured), and applies a
fine-tuned model for secondary structure prediction.

Author: Ayla Chin
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model cache to avoid reloading models multiple times
_model_cache = {}

def clear_model_cache():
    """Clear the model cache to free up memory."""
    global _model_cache
    _model_cache.clear()
    print("🗑️ Model cache cleared")

def get_cache_info():
    """Get information about cached models."""
    return {
        "cached_models": list(_model_cache.keys()),
        "cache_size": len(_model_cache)
    }


def clean_sequence(seq):
    lines = seq.strip().splitlines()
    if lines and lines[0].startswith(">"):
        lines = lines[1:]
    return "".join("".join(lines).split()).upper()

def get_fasta_source_and_accession(fasta_file, override=None):
    """
    Extracts accession, sequence, source classification, and detected source from a FASTA file.
    """
    accession, sequence, detected_source = get_accession_and_sequence(fasta_file)
    source = override if override else classify_fasta_source(accession)
    return accession, sequence, source, detected_source

# Sliding window function to segment the sequence
def sliding_window(sequence, window_size=512, num_segments=10):
    seq_length = len(sequence)
    if seq_length < window_size:
        return [sequence]
    step_size = max(1, (seq_length - window_size) // (num_segments - 1))
    segments = [sequence[i : i + window_size] for i in range(0, seq_length - window_size + 1, step_size)]
    return segments[:num_segments]

#Sliding window applied to the prediction model
def sliding_window_prediction(sequence, model, tokenizer, window_size=512, num_segments=10):
    label_map = {0: "H", 1: "B", 2: "C"}
    segments = sliding_window(sequence, window_size, num_segments)
    predictions = []

    for segment in segments:
        tokenized_inputs = tokenizer(" ".join(segment),
                                     padding="max_length",
                                     truncation=True,
                                     max_length=window_size,
                                     return_tensors="pt")
        input_ids = tokenized_inputs["input_ids"].to(device)
        attention_mask = tokenized_inputs["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
            pred_labels = torch.argmax(logits, dim=2).cpu().numpy().flatten()
        
        predictions.append("".join([label_map[p] for p in pred_labels[:len(segment)]]))

    if not predictions:
        return "C" * len(sequence)

    merged_prediction = predictions[0]
    for i in range(1, len(predictions)):
        merged_prediction += predictions[i][-(window_size // 2):]
    return merged_prediction[:len(sequence)]

def extract_sequence_from_fasta(fasta_file):
    with open(fasta_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            return "".join(str(record.seq).strip().split()).upper()


def extract_protein_name(input_data):
    """
    Extract protein name from either a FASTA file path or FASTA content string

    Args:
        input_data: Either a file path to FASTA file or FASTA content string

    Returns:
        str: Protein name or "Unknown" if not found
    """
    protein_name = "Unknown"

    try:
        # Check if input_data is a file path
        if os.path.isfile(input_data):
            # It's a file path
            with open(input_data, "r") as f:
                for record in SeqIO.parse(f, "fasta"):
                    description = record.description
                    protein_name = description.split(None, 1)[1] if len(description.split(None, 1)) > 1 else "Unknown"
                    break  # Take first record
        else:
            # It's FASTA content string
            lines = input_data.strip().split('\n')
            if lines and lines[0].startswith('>'):
                # Extract protein name from FASTA header
                header = lines[0][1:]  # Remove '>' character
                parts = header.split(None, 1)
                protein_name = parts[1] if len(parts) > 1 else "Unknown"
            else:
                # No FASTA header, just sequence
                protein_name = "Unknown"
    except Exception as e:
        print(f"⚠️ Warning: Could not extract protein name: {e}")
        protein_name = "Unknown"

    return protein_name

def fetch_alphafold_secondary_structure(accession):
    pdb_data = fetch_alphafold_pdb(accession)
    if pdb_data:
        print("✅ Successfully retrieved AlphaFold PDB data.")
    else:
        print("❌ Error: Failed to retrieve AlphaFold PDB data.")
        return None

    pdb_filename = f"pdb_files/{accession}.pdb"
    save_pdb_file(accession, pdb_data)
    residues, secondary_struct = get_dssp_secondary_structure(pdb_filename)
    smoothed_structure, mask = smooth_secondary_structure(secondary_struct)

    if not residues or not smoothed_structure:
        print("❌ Error: Failed to extract DSSP secondary structure.")
        return None
    else:
        return "".join(smoothed_structure)
        


def classify_fasta_source(accession):
    detected_source = "unknown"

    if "|" in accession:
        #print("🔍 '|' found in accession")
        parts = accession.split("|")

        # Case: UniProt FASTA header (sp|P01308|INS_HUMAN)
        if len(parts) > 1 and (parts[0].startswith(">sp") or parts[0].startswith(">tr")):
            detected_source = "uniprot"

        # Case: PDB-like chain-labeled headers (e.g., 9GA8_1|Chain)
        elif len(parts[0].split("_")[0]) == 4 and parts[0].split("_")[0].isalnum():
            detected_source = "pdb"

    # Also check non-pipe-formatted strings
    elif len(accession.split("_")[0]) == 4 and accession.split("_")[0].isalnum():
        print("PDB checked through here")
        detected_source = "pdb"

    elif accession.startswith("gi|") or "ref" in accession:
        detected_source = "ncbi"

    else:
        detected_source = "uniprot"

    print(f"🔍 Detected source: {detected_source} for accession: {accession}")
    return detected_source



def get_pdb_data(accession):
    if classify_fasta_source(accession) == "pdb":
        print("🔗 Detected PDB ID(s)...")
        accession= accession.split("|")[0][:4].upper()  # Strip Chain info if present
        url = f"https://files.rcsb.org/download/{accession}.pdb"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"❌ Error: Failed to fetch PDB structure from RCSB for {accession}. Status: {response.status_code}")
            return None

        pdb_dir = platform_manager.app_root / "pdb_files"
        pdb_dir.mkdir(exist_ok=True)
        pdb_filename = str(pdb_dir / f"{accession}.pdb")
        with open(pdb_filename, "w") as f:
            f.write(response.text)

        print(f"✅ Successfully saved {accession}.pdb")
        pdb_residues, secondary_struct = get_dssp_secondary_structure(pdb_filename)
        smoothed_structure, mask = smooth_secondary_structure(secondary_struct)

        if not pdb_residues or not smoothed_structure:
            print("❌ Error: Failed to extract DSSP secondary structure.")
            return None
        else:
            return "".join(smoothed_structure), pdb_residues


def get_uniprot_data(accession):
    if classify_fasta_source(accession) == "uniprot":
        print("🔗 Detected UniProt accession...")
        pdb_ids = get_pdb_ids_from_uniprot(accession)
        if not pdb_ids:
            print(f"❌ Error: No PDB IDs found for UniProt accession {accession}.")
            return None
        print(f"✅ Found PDB IDs: {', '.join(pdb_ids)}")
        
        # Fetch the first PDB structure
        pdb_data = fetch_pdb_structure(pdb_ids[0])
        if not pdb_data:
            print(f"❌ Error: Failed to fetch PDB structure for {pdb_ids[0]}.")
            return None
        print(f"✅ Successfully retrieved PDB data for {pdb_ids[0]}.")
        
        pdb_filename = save_pdb_file(pdb_ids[0], pdb_data)
        pdb_residues, secondary_struct = get_dssp_secondary_structure(pdb_filename)
        #print(f" extracted secondary struct from PDB and dssp {secondary_struct}")
        smoothed_structure, mask = smooth_secondary_structure(secondary_struct)

        if not pdb_residues or not smoothed_structure:
            print("❌ Error: Failed to extract DSSP secondary structure.")
            return None
        else:
            return "".join(smoothed_structure), pdb_residues
        

def get_ncbi_data(accession):
    if classify_fasta_source(accession) == "ncbi":
        print("🔗 Detected NCBI accession...")
        # For NCBI, we can use the accession directly to fetch PDB data
        pdb_data = fetch_pdb_structure(accession)
        if not pdb_data:
            print(f"❌ Error: Failed to fetch PDB structure for {accession}.")
            return None
        print(f"✅ Successfully retrieved PDB data for {accession}.")
        
        pdb_filename = save_pdb_file(accession, pdb_data)
        pdb_residues, secondary_struct = get_dssp_secondary_structure(pdb_filename)
        smoothed_structure, mask = smooth_secondary_structure(secondary_struct)

        if not pdb_residues or not smoothed_structure:
            print("❌ Error: Failed to extract DSSP secondary structure.")
            return None
        else:
            return "".join(smoothed_structure), pdb_residues


def display_aligned_sequences(primary_seq, secondary_struct, block_size=50):
    """
    Displays the primary sequence and its secondary structure in aligned blocks.
    """
    print("\nPrimary Sequence and Secondary Structure (Aligned Format):\n")

    for i in range(0, len(primary_seq), block_size):
        chunk_seq = primary_seq[i:i+block_size]
        chunk_struct = secondary_struct[i:i+block_size]

        seq_line = f"SEQ {i+1:<5} {chunk_seq}  {min(i+block_size, len(primary_seq))}"
        str_line = f"STR {' ' * 6} {chunk_struct}"

        print(seq_line)
        print(str_line)
        print("-" * max(len(seq_line), len(str_line)))  # optional separator


def load_model_and_tokenizer(model_dir):
    """
    Load model and tokenizer with caching to avoid reloading.

    Args:
        model_dir: Path to model directory

    Returns:
        tuple: (model, tokenizer)
    """
    # Convert to string for consistent cache key
    model_dir_str = str(Path(model_dir).resolve())

    # First check if model is preloaded
    try:
        from model_preloader import get_preloaded_model
        preloaded = get_preloaded_model(model_dir_str)
        if preloaded:
            print(f"⚡ Using preloaded model: {model_dir}")
            return preloaded
    except ImportError:
        # Preloader not available, continue with normal loading
        pass

    # Check if model is already cached
    if model_dir_str in _model_cache:
        print(f"📦 Using cached model: {model_dir}")
        return _model_cache[model_dir_str]

    # Load model if not cached
    model_dir = Path(model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

    print(f"📦 Loading model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=False)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=False, use_safetensors=True)
    model.to(device)

    # Cache the loaded model and tokenizer
    _model_cache[model_dir_str] = (model, tokenizer)
    print(f"✅ Model loaded and cached: {model_dir}")

    return model, tokenizer


def overlay_structures(model_pred, dssp_residues, dssp_structs, sequence):
    overlaid = list(model_pred)
    dssp_len = len(dssp_residues)
    source_mask = ["M"] * len(model_pred)


    # Replace model prediction with DSSP values where they match the input sequence
    i = 0  # index for model_pred
    j = 0  # index for dssp_residues
    while i < len(sequence) and j < dssp_len:
        if sequence[i] == dssp_residues[j]:
            overlaid[i] = dssp_structs[j]
            source_mask[i]= "D"
            j += 1
        i += 1

    return "".join(overlaid), source_mask


def compute_structure_percent(structure):
    counts = {"H": 0, "B": 0, "C": 0}
    for char in structure:
        if char in counts:
            counts[char] += 1
    total = sum(counts.values())
    return {k: (v / total) * 100 for k, v in counts.items()}



# Global progress callback for GUI integration
_progress_callback = None

def set_progress_callback(callback):
    """Set the progress callback function for GUI integration"""
    global _progress_callback
    _progress_callback = callback

def update_workflow_progress(step, total_steps, message):
    """Update progress if callback is set"""
    if _progress_callback:
        _progress_callback(step, total_steps, message)

def run_final_workflow(sequence, accession=None):
    print("\n🔬 Running full workflow...")
    start_time = time.time()

    # Extract protein name from original input (may contain FASTA header)
    protein_name = extract_protein_name(sequence)

    # Clean the sequence to remove FASTA headers and get pure amino acid sequence
    clean_seq = clean_sequence(sequence)
    print(f"📝 Cleaned sequence length: {len(clean_seq)} amino acids")

    #Step 1: Classify protein type and pivot to model prediction
    update_workflow_progress(1, 7, "Classifying protein type")
    # Call classify_sequence only once and unpack all results (use cleaned sequence)
    disorder_result, membrane_class, tm_regions, class_label = classify_sequence(clean_seq)
    protein_disorder = disorder_result
    protein_membrane = membrane_class

    update_workflow_progress(2, 7, "Loading prediction model")
    if class_label == "membrane":
        model_path = get_model_path("membrane")
    elif class_label == "disordered":
        model_path = get_model_path("disordered")
    else:
        model_path = get_model_path("structured")
    print(f"📦 Using model for '{class_label}': {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)

    update_workflow_progress(3, 7, "Generating structure predictions")
    model_pred = sliding_window_prediction(clean_seq, model, tokenizer)
    model_pred_list, _ = smooth_secondary_structure(list(model_pred))  # discard mask
    model_pred = "".join(model_pred_list)   
    model_used = class_label




    #Step 2: Start BLAST search asynchronously (non-blocking)
    update_workflow_progress(4, 7, "Searching for similar structures")
    blast_results = {"hits": [], "completed": False, "success": False}

    def blast_callback(success, results):
        """Callback function for when BLAST completes"""
        blast_results["completed"] = True
        blast_results["success"] = success
        blast_results["hits"] = results if success else []
        if success and results:
            print(f"🔍 BLAST found {len(results)} similar structures")
        else:
            print("🔍 BLAST completed - no similar structures found")

    try:
        searcher = PDBBlastSearcher()
        searcher.setup_database
        query_file = "temp_query.fasta"
        with open(query_file, "w") as f:
            f.write(f">query\n{clean_seq}\n")
        output_prefix = "blast_output"

        # Start BLAST in background thread
        blast_thread = searcher.run_blast_search_threaded(
            query_file, output_prefix + ".txt", callback=blast_callback
        )
        print("🚀 BLAST search started in background...")

    except Exception as e:
        print(f"⚠️ BLAST search failed to start: {e}")
        print("   Continuing with model-only prediction...")
        blast_results["completed"] = True
        blast_results["success"] = False

    # Wait for BLAST to complete (with timeout) while doing other work
    blast_wait_time = 0
    max_blast_wait = 30  # Maximum 30 seconds to wait for BLAST

    # Check BLAST results periodically
    while not blast_results["completed"] and blast_wait_time < max_blast_wait:
        time.sleep(1)
        blast_wait_time += 1

    # Get final BLAST results
    hits = blast_results["hits"] if blast_results["success"] else []

    if not hits:
        if blast_results["completed"]:
            print("⚠️ No BLAST hits found. Using model-only prediction...")
        else:
            print("⚠️ BLAST search timed out. Using model-only prediction...")

        accession = "Unknown"
        identity = 0.0
        e_value = str('N/A')
        dssp_residues = ""
        dssp_structs = ""
        hits = [{"subject_id": "Unknown", "percent_identity": 0.0, "evalue": str('N/A')}]
    else:
        accession = hits[0]['subject_id'][:4]
        identity = hits[0]['percent_identity']
        e_value = hits[0]['evalue']
        print(f"✅ Found matching accession: {accession} ({identity:.2f}% identity)")

        #Step 3: Get the PDB file based on the accession hit
        update_workflow_progress(5, 7, "Fetching experimental structure data")
        pdb_data = fetch_pdb_structure(accession)
        if pdb_data:
            pdb_file_path = save_pdb_file(accession, pdb_data)
            print(f"✅ PDB file saved: {pdb_file_path}")

            #Step 4: Extract secondary structure from the PDB file using DSSP
            pdb_residues, dssp_structs = get_dssp_secondary_structure(pdb_file_path)
            if dssp_structs and pdb_residues:
                print(f"✅ DSSP secondary structure extracted: {len(dssp_structs)} residues")
                dssp_structs_list, _ = smooth_secondary_structure(list(dssp_structs))
                dssp_structs = "".join(dssp_structs_list)
                dssp_residues = "".join(pdb_residues)
            else:
                print("⚠️ Could not extract DSSP data. Using model-only prediction...")
                dssp_residues = ""
                dssp_structs = ""
        else:
            print("⚠️ Could not fetch PDB data. Using model-only prediction...")
            dssp_residues = ""
            dssp_structs = ""




    #Step 5: Overlay the model prediction with the DSSP structure
    update_workflow_progress(6, 7, "Combining predictions with experimental data")
    print("Here is the comparison of the model prediction and the DSSP structure:")
    display_aligned_sequences(clean_seq, model_pred)
    display_aligned_sequences(dssp_residues, dssp_structs)
    print("Now we will overlay the model prediction with the DSSP structure.")

    predicted_structure, overlay_mask = overlay_structures(model_pred, dssp_residues, dssp_structs, clean_seq)
    print(type(predicted_structure))
    predicted_structure_list, smooth_mask = smooth_secondary_structure(list(predicted_structure))
    predicted_structure_smoothed = "".join(predicted_structure_list)


    final_mask =[]
    for d,s in zip(overlay_mask, smooth_mask):
        if d == "D":
            final_mask.append("D")
        elif s == "S":
            final_mask.append("S")
        else:
            final_mask.append("M")

    predicted_structure_smoothed = "".join(predicted_structure_smoothed)

    # --- Coverage: strictly overlay only ---
    overlay_count = sum(1 for m in final_mask if m == "D")
    seq_len = len(clean_seq)

    if seq_len > 0 and overlay_count>1: # Only compute if BLAST found something
        coverage = (overlay_count / seq_len) * 100.0
    else:
        coverage = 0.0





    print("✅ Successfully overlaid model prediction with DSSP structure.")
    display_aligned_sequences(clean_seq, predicted_structure_smoothed)

    update_workflow_progress(7, 7, "🎉 Analysis complete!")
    end_time = time.time()

    return {
        "protein_name": protein_name,
        "disorder": protein_disorder,
        "membrane": protein_membrane,
        "tm_regions": tm_regions,
        "classification": class_label,
        "top hit": hits[0],
        "identity": identity,
        "residue_number": len(clean_seq),
        "e_value": e_value,
        "sequence": clean_seq,
        "accession": accession,
        "dssp": dssp_residues,
        "model": model_pred,
        "combined structure": predicted_structure_smoothed,
        "structure mask": final_mask,
        "model_used": model_used,
        "%helix": compute_structure_percent(predicted_structure_smoothed)['H'],
        "%sheet": compute_structure_percent(predicted_structure_smoothed)['B'],
        "%coil": compute_structure_percent(predicted_structure_smoothed)['C'],
        "time": f"{end_time - start_time:.2f} seconds",
        "coverage": coverage
    }

    


def main(fasta_file=None, sequence=None, source_override=None):
    print("\nProtein Secondary Structure Prediction")

    if fasta_file:
        if not os.path.exists(fasta_file):
            print("❌ Error: File not found!")
            return

        # Auto-detect source if override not giv
        classified_source = classify_fasta_source(fasta_file)
    

        print(f"\n✅ Extracted accession: {accession} (Detected source: {classified_source}, Used source: {source})")

        if source=="pdb":
            get_pdb_data(accession)

        elif source=="uniprot":
            get_uniprot_data(accession)
        elif source=="ncbi":
            get_ncbi_data(accession)
        else:
            get_pdb_accession_by_sequence(sequence, identity_threshold=80.0)
        if not sequence:
            sequence = extract_sequence_from_fasta(fasta_file)
            if not sequence:
                print("❌ Error: No sequence found in FASTA file.")
                return


    elif sequence:
        print("\n🧬 Using direct sequence input.")
        accession = None
        source = None
    else:
        print("❌ Error: No input provided.")
        return
    protein_name= extract_protein_name(fasta_file) if fasta_file else "Unknown Protein"
    result = run_final_workflow(sequence, accession)
    if not result:
        print("❌ Workflow failed.")
        return
    print(f"\n Protein Name: {protein_name}")
    print(f"\n✅ Accession: {result['accession']} ({result['identity']:.2f}%)")
    print(f"📦 Model Used: {result['model_used']}")
    display_aligned_sequences(clean_sequence(result['sequence']), result['combined structure'])
    print(f"⏱️ Time elapsed: {result['time']}")




if __name__ == "__main__":
    print("1. Enter a protein sequence")
    print("2. Upload a FASTA file")
    choice = input("\nChoose an option (1 or 2): ").strip()

    if choice == "1":
        seq = input("Enter protein sequence: ").strip()
        main(sequence=seq)
    elif choice == "2":
        fpath = input("Enter path to FASTA file: ").strip()
    else:
        print("❌ Invalid choice. Exiting.")
        
