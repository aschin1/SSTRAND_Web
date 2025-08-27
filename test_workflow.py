import unittest
from unittest.mock import patch, MagicMock
from final_workflow4 import run_final_workflow
import os
import torch

# Create dummy directories for testing
test_models_path = os.path.join(os.getcwd(), 'test_models')
structured_path = os.path.join(test_models_path, 'structured')
disordered_path = os.path.join(test_models_path, 'disordered')
membrane_path = os.path.join(test_models_path, 'membrane')

os.makedirs(structured_path, exist_ok=True)
os.makedirs(disordered_path, exist_ok=True)
os.makedirs(membrane_path, exist_ok=True)

class TestProteinWorkflow(unittest.TestCase):
    
    @patch('final_workflow4.load_model_and_tokenizer')
    @patch('final_workflow4.classify_sequence')
    @patch('final_workflow4.PDBBlastSearcher')
    @patch('final_workflow4.fetch_pdb_structure')
    @patch('final_workflow4.get_dssp_secondary_structure')
    @patch('final_workflow4.display_aligned_sequences')
    @patch('final_workflow4.extract_protein_name')
    @patch('final_workflow4.save_pdb_file')
    @patch('final_workflow4.get_model_path')
    def test_full_successful_workflow(self, mock_get_model_path, mock_save_pdb, mock_protein_name, mock_display, mock_dssp, mock_pdb_fetch, mock_blast_searcher, mock_classify, mock_load_model):
        """Test a full successful run of the workflow."""
        
        # --- FIX: Mock the model's output correctly ---
        mock_model_output = MagicMock()
        mock_model_output.logits = torch.randn(1, 512, 3) # Simulates a tensor output
        mock_model = MagicMock()
        mock_model.return_value = mock_model_output
        mock_load_model.return_value = (mock_model, MagicMock())
        # --- End of fix ---

        # Set up other mocks for a successful scenario
        mock_protein_name.return_value = "Test Protein"
        mock_classify.return_value = ({'disordered_regions': []}, 'non-membrane', [], 'structured')
        
        # Configure the BLAST mock to return a successful hit
        mock_blast_searcher.return_value.run_blast_search_threaded.side_effect = lambda *args, **kwargs: kwargs['callback'](True, [{'subject_id': '4XYZ', 'percent_identity': 95.5, 'evalue': 1e-10}])
        
        mock_pdb_fetch.return_value = "mock PDB data"
        mock_save_pdb.return_value = "/dummy/path/4XYZ.pdb"
        mock_dssp.return_value = (list("DFCMG"), list("HHHHH"))
        mock_get_model_path.return_value = structured_path

        test_sequence = "DFCMGSMRVAALLYYAAPSGRPHSSWRPGKGLDAPQHCTGMPRGSVECPV"
        
        result = run_final_workflow(test_sequence)
        
        # Assertions to check the result dictionary
        self.assertIsNotNone(result)
        self.assertEqual(result['classification'], 'structured')
        self.assertEqual(result['accession'], '4XYZ')
        self.assertEqual(result['identity'], 95.5)
        self.assertEqual(result['e_value'], 1e-10)
        self.assertAlmostEqual(result['coverage'], (5/len(test_sequence)) * 100, places=2)
        
    @patch('final_workflow4.load_model_and_tokenizer')
    @patch('final_workflow4.classify_sequence')
    @patch('final_workflow4.PDBBlastSearcher')
    @patch('final_workflow4.fetch_pdb_structure')
    @patch('final_workflow4.get_model_path')
    @patch('final_workflow4.extract_protein_name')
    @patch('final_workflow4.display_aligned_sequences')
    def test_blast_failure(self, mock_display, mock_protein_name, mock_get_model_path, mock_pdb_fetch, mock_blast_searcher, mock_classify, mock_load_model):
        """Test the workflow when BLAST returns no hits."""
        
        # --- FIX: Mock the model's output correctly ---
        mock_model_output = MagicMock()
        mock_model_output.logits = torch.randn(1, 512, 3)
        mock_model = MagicMock()
        mock_model.return_value = mock_model_output
        mock_load_model.return_value = (mock_model, MagicMock())
        # --- End of fix ---
        
        mock_protein_name.return_value = "Test Protein"
        mock_classify.return_value = ({'disordered_regions': []}, 'non-membrane', [], 'structured')
        mock_get_model_path.return_value = structured_path

        # Configure the BLAST mock to return a failure (no hits)
        mock_blast_searcher.return_value.run_blast_search_threaded.side_effect = lambda *args, **kwargs: kwargs['callback'](True, [])
        
        test_sequence = "DFCMGSMRVAALLYYAAPSGRPHSSWRPGKGLDAPQHCTGMPRGSVECPV"
        
        result = run_final_workflow(test_sequence)
        
        # Assertions to check that BLAST-related fields are handled gracefully
        self.assertEqual(result['accession'], 'Unknown')
        self.assertEqual(result['identity'], 0.0)
        self.assertEqual(result['e_value'], 'N/A')
        self.assertEqual(result['coverage'], 0.0)
        self.assertEqual(result['classification'], 'structured')

    @patch('final_workflow4.load_model_and_tokenizer')
    @patch('final_workflow4.classify_sequence')
    @patch('final_workflow4.PDBBlastSearcher')
    @patch('final_workflow4.fetch_pdb_structure')
    @patch('final_workflow4.get_model_path')
    @patch('final_workflow4.extract_protein_name')
    @patch('final_workflow4.save_pdb_file')
    @patch('final_workflow4.display_aligned_sequences')
    def test_dssp_extraction_failure(self, mock_display, mock_save_pdb, mock_protein_name, mock_get_model_path, mock_pdb_fetch, mock_blast_searcher, mock_classify, mock_load_model):
        """Test the workflow when DSSP extraction fails, forcing model-only prediction."""
        
        # --- FIX: Mock the model's output correctly ---
        mock_model_output = MagicMock()
        mock_model_output.logits = torch.randn(1, 512, 3)
        mock_model = MagicMock()
        mock_model.return_value = mock_model_output
        mock_load_model.return_value = (mock_model, MagicMock())
        # --- End of fix ---
        
        mock_protein_name.return_value = "Test Protein"
        mock_classify.return_value = ({'disordered_regions': []}, 'non-membrane', [], 'structured')
        mock_get_model_path.return_value = structured_path

        # Mock a successful BLAST hit
        mock_blast_searcher.return_value.run_blast_search_threaded.side_effect = lambda *args, **kwargs: kwargs['callback'](True, [{'subject_id': '4XYZ', 'percent_identity': 95.5, 'evalue': 1e-10}])
        
        mock_pdb_fetch.return_value = "mock PDB data"
        mock_save_pdb.return_value = "/dummy/path/4XYZ.pdb"
        
        # Mock DSSP to return no data
        with patch('final_workflow4.get_dssp_secondary_structure', return_value=(None, None)):
            test_sequence = "DFCMGSMRVAALLYYAAPSGRPHSSWRPGKGLDAPQHCTGMPRGSVECPV"
            result = run_final_workflow(test_sequence)
        
            # Assertions to check that DSSP-related fields are empty and coverage is 0
            self.assertEqual(result['dssp'], "")
            self.assertAlmostEqual(result['coverage'], 0.0, places=2)
            self.assertEqual(result['accession'], '4XYZ')
            self.assertEqual(result['identity'], 95.5)

    @patch('final_workflow4.os.path.exists', return_value=False)
    def test_model_path_not_found(self, mock_exists):
        """Test handling of a FileNotFoundError when a model path does not exist."""
        from final_workflow4 import load_model_and_tokenizer
        with self.assertRaises(FileNotFoundError):
            load_model_and_tokenizer('/non/existent/path')
            
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)