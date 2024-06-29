from tokenizers import ByteLevelBPETokenizer, normalizers
from tokenizers.normalizers import NFKC, Lowercase
import pandas as pd
data = pd.read_csv("./NLP/Sentiment Analysis/util/IMDB Dataset.csv").to_numpy()
corpus = data[:, 0]
tokenizer = ByteLevelBPETokenizer()
tokenizer.normalizer = normalizers.Sequence(
    [NFKC(), Lowercase(), normalizers.Replace('<br />', ' ')])
tokenizer.train_from_iterator(corpus, vocab_size=30000, min_frequency=2)
tokenizer.save('./NLP/Sentiment Analysis/tokenizer/tokenizer.json')
