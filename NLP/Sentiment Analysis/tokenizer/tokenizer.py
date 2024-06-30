from tokenizers import ByteLevelBPETokenizer, normalizers
from tokenizers.normalizers import NFKC, Lowercase
import pandas as pd
data = pd.read_csv("./NLP/Sentiment Analysis/util/IMDB Dataset.csv").to_numpy()
corpus = data[:, 0]
tokenizer = ByteLevelBPETokenizer()
remove_urls = r'https://\S+|www\.\S+'
remove_html = r'<.*?>'
remove_punct = r'[^\w\s]'
remove_emojis = ("["
                 u"\U0001F600-\U0001F64F"  # emoticons
                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                 u"\U00002702-\U000027B0"
                 u"\U000024C2-\U0001F251"
                 "]+")
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.Replace(remove_urls, ''),
     normalizers.Replace(remove_html, ''),
     normalizers.Replace(remove_punct, ''),
     normalizers.Replace(remove_emojis, ''),
     Lowercase()])
tokenizer.train_from_iterator(corpus, vocab_size=30000, min_frequency=2)
tokenizer.save('./NLP/Sentiment Analysis/tokenizer/tokenizer.json')
