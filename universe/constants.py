"""
Written by: Akintunde 'theyorubayesian' Oladipo
14/Nov/2021
"""

# X, U, B, O and Z are mapped to 0
ACID_MAP = {
    "L": 1, "A": 2, "V": 3, "G": 4, 
    "E": 5, "S": 6, "I": 7, "R": 8, 
    "D": 9, "K": 10, "T": 11, "P": 12, 
    "F": 13, "N": 14, "Q": 15, "Y": 16,
    "M": 17, "H": 18, "C": 19, "W": 20, 
}

ACID_MAP_INV = {v: k for k, v in ACID_MAP.items()}

PADDING_VALUE = 21

VOCAB_SIZE = 22