def read_sham_file(filename):
    """
    Parses a .sham file containing genome reads.

    Parameters:
        filename (str): Path to the input .sham file. Each line must be in the format: <start_pos> <TAB> <read_string>

    Returns:
        tuple:
            - observations (list): List of tuples (start_position, read_string)
            - max_pos (int): Highest genome position observed across all reads
    """
    observations = []
    max_pos = 0

    with open(filename, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                start_str, read = line.split('\t')
                start = int(start_str.strip())
                read = read.strip()
                observations.append((start, read))
                max_pos = max(max_pos, start + len(read))
            except ValueError:
                raise ValueError(f"Line {line_number} in '{filename}' is malformed: '{line}'")

    return observations, max_pos
