import pytest
import os
import tempfile
from parser import read_sham_file


@pytest.fixture
def temp_sham_file():
    """Create a temporary .sham file for testing."""
    # Create a temporary file
    fd, filepath = tempfile.mkstemp(suffix='.sham')
    os.close(fd)
    
    # Return the filepath for use in tests
    yield filepath
    
    # Clean up after tests
    os.unlink(filepath)


def test_empty_file(temp_sham_file):
    """Test parsing an empty .sham file."""
    # Create an empty file
    with open(temp_sham_file, 'w') as f:
        pass
    
    # Parse the file
    observations, max_pos = read_sham_file(temp_sham_file)
    
    # Check results
    assert len(observations) == 0
    assert max_pos == 0


def test_single_read(temp_sham_file):
    """Test parsing a .sham file with a single read."""
    # Create a file with one read
    with open(temp_sham_file, 'w') as f:
        f.write("5\t10101\n")
    
    # Parse the file
    observations, max_pos = read_sham_file(temp_sham_file)
    
    # Check results
    assert len(observations) == 1
    assert observations[0] == (5, "10101")
    assert max_pos == 10  # position 5 + length 5


def test_multiple_reads(temp_sham_file):
    """Test parsing a .sham file with multiple reads."""
    # Create a file with multiple reads
    with open(temp_sham_file, 'w') as f:
        f.write("0\t101\n")
        f.write("5\t010\n")
        f.write("10\t11111\n")
    
    # Parse the file
    observations, max_pos = read_sham_file(temp_sham_file)
    
    # Check results
    assert len(observations) == 3
    assert observations[0] == (0, "101")
    assert observations[1] == (5, "010")
    assert observations[2] == (10, "11111")
    assert max_pos == 15  # position 10 + length 5


def test_empty_lines(temp_sham_file):
    """Test parsing a .sham file with empty lines."""
    # Create a file with empty lines
    with open(temp_sham_file, 'w') as f:
        f.write("\n")
        f.write("0\t101\n")
        f.write("\n")
        f.write("5\t010\n")
        f.write("\n")
    
    # Parse the file
    observations, max_pos = read_sham_file(temp_sham_file)
    
    # Check results
    assert len(observations) == 2
    assert observations[0] == (0, "101")
    assert observations[1] == (5, "010")
    assert max_pos == 8  # position 5 + length 3


def test_whitespace_handling(temp_sham_file):
    """Test parsing a .sham file with extra whitespace."""
    # Create a file with extra whitespace
    with open(temp_sham_file, 'w') as f:
        f.write("  0  \t  101  \n")
        f.write("5\t010\n")
    
    # Parse the file
    observations, max_pos = read_sham_file(temp_sham_file)
    
    # Check results
    assert len(observations) == 2
    assert observations[0] == (0, "101")
    assert observations[1] == (5, "010")
    assert max_pos == 8  # position 5 + length 3


def test_malformed_line(temp_sham_file):
    """Test parsing a .sham file with a malformed line."""
    # Create a file with a malformed line
    with open(temp_sham_file, 'w') as f:
        f.write("0\t101\n")
        f.write("invalid line\n")
        f.write("5\t010\n")
    
    # Parse the file - should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        read_sham_file(temp_sham_file)
    
    # Check error message
    assert "malformed" in str(excinfo.value)
    assert "Line 2" in str(excinfo.value)


def test_non_integer_position(temp_sham_file):
    """Test parsing a .sham file with a non-integer position."""
    # Create a file with a non-integer position
    with open(temp_sham_file, 'w') as f:
        f.write("0\t101\n")
        f.write("abc\t010\n")
    
    # Parse the file - should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        read_sham_file(temp_sham_file)
    
    # Check error message
    assert "malformed" in str(excinfo.value)
    assert "Line 2" in str(excinfo.value)


def test_negative_position(temp_sham_file):
    """Test parsing a .sham file with a negative position."""
    # Create a file with a negative position
    with open(temp_sham_file, 'w') as f:
        f.write("-5\t101\n")
    
    # Parse the file
    observations, max_pos = read_sham_file(temp_sham_file)
    
    # Check results - negative positions are allowed
    assert len(observations) == 1
    assert observations[0] == (-5, "101")
    # The max_pos calculation in read_sham_file takes max(max_pos, start + len(read))
    # So even with negative positions, max_pos will be at least 0
    assert max_pos >= 0


def test_zero_position(temp_sham_file):
    """Test parsing a .sham file with a zero position."""
    # Create a file with a zero position
    with open(temp_sham_file, 'w') as f:
        f.write("0\t101\n")
    
    # Parse the file
    observations, max_pos = read_sham_file(temp_sham_file)
    
    # Check results
    assert len(observations) == 1
    assert observations[0] == (0, "101")
    assert max_pos == 3  # position 0 + length 3


def test_real_world_example(temp_sham_file):
    """Test parsing a more realistic .sham file."""
    # Create a file with a more realistic example
    with open(temp_sham_file, 'w') as f:
        f.write("0\t10101010101\n")
        f.write("10\t01010101010\n")
        f.write("20\t11111111111\n")
        f.write("30\t00000000000\n")
        f.write("40\t10101010101\n")
    
    # Parse the file
    observations, max_pos = read_sham_file(temp_sham_file)
    
    # Check results
    assert len(observations) == 5
    assert observations[0] == (0, "10101010101")
    assert observations[1] == (10, "01010101010")
    assert observations[2] == (20, "11111111111")
    assert observations[3] == (30, "00000000000")
    assert observations[4] == (40, "10101010101")
    assert max_pos == 51  # position 40 + length 11


@pytest.mark.parametrize("content,expected_max", [
    ("0\t1", 1),
    ("5\t1", 6),
    ("0\t1\n10\t1", 11),
    ("10\t1\n0\t1", 11),
    ("5\t11111\n20\t1\n10\t111", 21)
])
def test_max_position_calculation(temp_sham_file, content, expected_max):
    """Test max position calculation with various inputs."""
    # Create a file with the given content
    with open(temp_sham_file, 'w') as f:
        f.write(content)
    
    # Parse the file
    _, max_pos = read_sham_file(temp_sham_file)
    
    # Check max_pos
    assert max_pos == expected_max
