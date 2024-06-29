from tokenizers import ByteLevelBPETokenizer, normalizers
from tokenizers.normalizers import NFKC, Lowercase
from pathlib import Path
import pandas as pd
import numpy as np

paths = [str(x) for x in Path('./movie_data').glob('*.txt')]
tokenizer = ByteLevelBPETokenizer()
tokenizer.normalizer = normalizers.Sequence(
    [NFKC(), Lowercase(), normalizers.Replace('<br />', ' ')])
tokenizer.train(files=paths, vocab_size=30000, min_frequency=2)
tokenizer.save('tokenizer.json')
