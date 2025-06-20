# This script is intended for internal testing purposes to simulate large.sham files 

import random

def generate_sham(filename, genome_size=1000000, num_reads=500000, max_read_len=1000):
    with open(filename, 'w') as f:
        for _ in range(num_reads):
            start = random.randint(0, genome_size - max_read_len)
            read_len = random.randint(50, max_read_len)
            read = ''.join(random.choice('01') for _ in range(read_len))
            f.write(f"{start}\t{read}\n")

if __name__ == "__main__":
    generate_sham("large_example.sham")
