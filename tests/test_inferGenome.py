import pytest
import os
import tempfile
import sys
from unittest.mock import patch
from inferGenome import parse_args, pad_read


@pytest.fixture
def temp_files():
    """Create temporary test files."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create paths for test files
        test_sham = os.path.join(tmp_dir, "test.sham")
        test_output = os.path.join(tmp_dir, "output.tsv")
        
        # Create a simple test sham file
        with open(test_sham, 'w') as f:
            f.write("1\t1001\n5\t11111")
        
        # Return the file paths
        yield {
            'sham': test_sham,
            'output': test_output,
            'dir': tmp_dir
        }


class TestPadRead:
    """Tests for the pad_read function."""
    
    def test_pad_shorter_read(self):
        """Test padding a read that's shorter than the target length."""
        assert pad_read("101", 5) == "10100"
    
    def test_trim_longer_read(self):
        """Test trimming a read that's longer than the target length."""
        assert pad_read("1010101", 5) == "10101"
    
    def test_exact_length(self):
        """Test a read that's already the target length."""
        assert pad_read("10101", 5) == "10101"
    
    @pytest.mark.parametrize("read,length,expected", [
        ("", 3, "000"),           # Empty read
        ("1", 1, "1"),            # Single character, exact length
        ("0000", 10, "0000000000"),  # Pad with many zeros
        ("11111", 3, "111"),      # Significant trimming
        ("101010", 6, "101010"),  # Exact match
    ])
    def test_various_cases(self, read, length, expected):
        """Test various padding/trimming scenarios."""
        assert pad_read(read, length) == expected


class TestParseArgs:
    """Tests for the argument parsing functionality."""
    
    @patch('sys.argv')
    def test_bayesian_args(self, mock_argv, temp_files):
        """Test parsing arguments for Bayesian inference."""
        # Set up mock command line arguments
        mock_argv.__getitem__.side_effect = lambda i: [
            'inferGenome.py',
            '--sham', temp_files['sham'],
            '--out', temp_files['output'],
            '--p01', '0.1',
            '--p10', '0.2',
            '--prior1', '0.5'
        ][i]
        
        # Parse the arguments
        args = parse_args()
        
        # Verify the parsed arguments
        assert args.sham == temp_files['sham']
        assert args.out == temp_files['output']
        assert args.p01 == 0.1
        assert args.p10 == 0.2
        assert args.prior1 == 0.5
        assert not args.use_nn
        assert not args.use_gpu
    
    @patch('sys.argv')
    def test_bayesian_with_gpu(self, mock_argv, temp_files):
        """Test parsing arguments for Bayesian inference with GPU acceleration."""
        # Set up mock command line arguments
        mock_argv.__getitem__.side_effect = lambda i: [
            'inferGenome.py',
            '--sham', temp_files['sham'],
            '--out', temp_files['output'],
            '--p01', '0.1',
            '--p10', '0.2',
            '--prior1', '0.5',
            '--use-gpu'
        ][i]
        
        # Parse the arguments
        args = parse_args()
        
        # Verify the parsed arguments
        assert args.sham == temp_files['sham']
        assert args.out == temp_files['output']
        assert args.p01 == 0.1
        assert args.p10 == 0.2
        assert args.prior1 == 0.5
        assert not args.use_nn
        assert args.use_gpu
    
    @patch('sys.argv')
    def test_neural_network_args(self, mock_argv, temp_files):
        """Test parsing arguments for Neural Network inference."""
        # Set up mock command line arguments
        mock_argv.__getitem__.side_effect = lambda i: [
            'inferGenome.py',
            '--sham', temp_files['sham'],
            '--out', temp_files['output'],
            '--use-nn',
            '--model', 'mymodel.pth'
        ][i]
        
        # Parse the arguments
        args = parse_args()
        
        # Verify the parsed arguments
        assert args.sham == temp_files['sham']
        assert args.out == temp_files['output']
        assert args.use_nn
        assert args.model == 'mymodel.pth'
    
    @patch('sys.argv')
    def test_default_output_file(self, mock_argv, temp_files):
        """Test that output file defaults to None when not specified."""
        # Set up mock command line arguments without output file
        mock_argv.__getitem__.side_effect = lambda i: [
            'inferGenome.py',
            '--sham', temp_files['sham'],
            '--p01', '0.1',
            '--p10', '0.2',
            '--prior1', '0.5'
        ][i]
        
        # Parse the arguments
        args = parse_args()
        
        # Verify the parsed arguments
        assert args.sham == temp_files['sham']
        assert args.out is None  # Should be None before main() sets default
    
    @patch('sys.argv')
    def test_default_model_path(self, mock_argv, temp_files):
        """Test that model path defaults to 'model.pth' when not specified."""
        # Set up mock command line arguments without model path
        mock_argv.__getitem__.side_effect = lambda i: [
            'inferGenome.py',
            '--sham', temp_files['sham'],
            '--out', temp_files['output'],
            '--use-nn'
        ][i]
        
        # Parse the arguments
        args = parse_args()
        
        # Verify the parsed arguments
        assert args.sham == temp_files['sham']
        assert args.out == temp_files['output']
        assert args.use_nn
        assert args.model == 'model.pth'  # Default model path


class TestMainFunction:
    """Tests for the main function."""
    
    @patch('inferGenome.compute_posteriors')
    @patch('inferGenome.read_sham_file')
    @patch('inferGenome.write_output')
    @patch('sys.argv')
    def test_bayesian_inference(self, mock_argv, mock_write, mock_read, mock_compute, temp_files):
        """Test main function with Bayesian inference."""
        # Set up mock command line arguments
        mock_argv.__getitem__.side_effect = lambda i: [
            'inferGenome.py',
            '--sham', temp_files['sham'],
            '--out', temp_files['output'],
            '--p01', '0.1',
            '--p10', '0.2',
            '--prior1', '0.5'
        ][i]
        
        # Mock read_sham_file return value
        mock_read.return_value = ([(1, '1001'), (5, '11111')], 10)
        
        # Mock compute_posteriors return value
        mock_compute.return_value = [0.5] * 11
        
        # Import main function and run it
        from inferGenome import main
        main()
        
        # Verify function calls
        mock_read.assert_called_once_with(temp_files['sham'])
        mock_compute.assert_called_once()
        mock_write.assert_called_once_with([0.5] * 11, temp_files['output'])
    
    @patch('torch.load')
    @patch('nn_inference.nn_infer')
    @patch('inferGenome.read_sham_file')
    @patch('inferGenome.write_output')
    @patch('sys.argv')
    def test_neural_network_inference(self, mock_argv, mock_write, mock_read, mock_nn_infer, mock_torch_load, temp_files):
        """Test main function with Neural Network inference."""
        # Set up mock command line arguments
        mock_argv.__getitem__.side_effect = lambda i: [
            'inferGenome.py',
            '--sham', temp_files['sham'],
            '--out', temp_files['output'],
            '--use-nn',
            '--model', 'mymodel.pth'
        ][i]
        
        # Mock read_sham_file return value
        mock_read.return_value = ([(1, '1001'), (5, '11111')], 10)
        
        # Mock torch.load return value
        mock_torch_load.return_value = {'input_dim': 10}
        
        # Mock nn_infer return value
        mock_nn_infer.return_value = [0.5] * 11
        
        # Import main function and run it
        from inferGenome import main
        main()
        
        # Verify function calls
        mock_read.assert_called_once_with(temp_files['sham'])
        mock_torch_load.assert_called_once_with('mymodel.pth', map_location='cpu')
        mock_nn_infer.assert_called_once()
        mock_write.assert_called_once_with([0.5] * 11, temp_files['output'])
    
    @patch('sys.argv')
    def test_missing_bayesian_params(self, mock_argv, temp_files):
        """Test that an error is raised when Bayesian parameters are missing."""
        # Set up mock command line arguments without Bayesian parameters
        mock_argv.__getitem__.side_effect = lambda i: [
            'inferGenome.py',
            '--sham', temp_files['sham'],
            '--out', temp_files['output']
        ][i]
        
        # Import main function
        from inferGenome import main
        
        # Verify that ValueError is raised
        with pytest.raises(ValueError) as excinfo:
            main()
        
        # Check error message
        assert "Bayesian mode requires" in str(excinfo.value)
    
    @patch('inferGenome.read_sham_file')
    @patch('inferGenome.write_output')
    @patch('inferGenome.compute_posteriors')
    @patch('sys.argv')
    def test_default_output_filename(self, mock_argv, mock_compute, mock_write, mock_read, temp_files):
        """Test that output filename defaults to 'output.tsv' when not specified."""
        # Set up mock command line arguments without output file
        mock_argv.__getitem__.side_effect = lambda i: [
            'inferGenome.py',
            '--sham', temp_files['sham'],
            '--p01', '0.1',
            '--p10', '0.2',
            '--prior1', '0.5'
        ][i]
        
        # Mock read_sham_file return value
        mock_read.return_value = ([(1, '1001'), (5, '11111')], 10)
        
        # Mock compute_posteriors return value
        mock_compute.return_value = [0.5] * 11
        
        # Import main function and run it
        from inferGenome import main
        main()
        
        # Verify write_output was called with default filename
        mock_write.assert_called_once_with([0.5] * 11, "output.tsv")
