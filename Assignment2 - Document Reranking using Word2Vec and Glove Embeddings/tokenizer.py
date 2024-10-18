import re
from nltk.tokenize import WordPunctTokenizer, TreebankWordTokenizer

class SimpleTokenizer:      # Tokenize and remove punctuation  

    def __init__(self):
        self._tokenizer_ = WordPunctTokenizer()

    def tokenize(self, text):
        tokens = self._tokenizer_.tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalnum()]
        return tokens

class SimpleTokenizer2:     # Tokenizes and removes punctuation but does not remove contractions

    def __init__(self):
        self._tokenizer_ = TreebankWordTokenizer()

    def tokenize(self, text):
        tokens = self._tokenizer_.tokenize(text)
        tokens = [token.lower() for token in tokens if not (len(token)==1 and not token.isalnum())]
        return tokens




# class SimpleTokenizer:
#     def __init__(self):
#         self.pattern = r"[\ ,\.:;\"'/\-%\(\)\[\]]+"     # Define the delimiter pattern for tokenization     
    
#     def tokenize(self, text):
#         tokens = re.split(self.pattern, text)
#         tokens = [token.lower() for token in tokens if token]   # Filter out any empty strings that might have resulted from splitting    
#         return tokens

# class SimpleTokenizer:

#     def __init__(self, delimiters):
#         self.delimiters = delimiters
#         pattern = "[" + re.escape(''.join(self.delimiters)) + "]+"
#         self._tokenizer_ = RegexpTokenizer(pattern=pattern, gaps=True)

#     def tokenize(self, text: str)->List[str]:
#         tokens = self._tokenizer_.tokenize(text.lower())
#         return tokens
    
# if __name__ == "__main__":
#     text = """This, is the 'text' I want to tokenize; My name is:chinmay. """
#     text2 = "He said: 'I'm 50% sure that it's 10:30 am - (early morning) on 03/05/2024; check [this]!"
#     tokenizer = SimpleTokenizer([" ", ",", ".", ":", ";", "\"", "\'", '/', '-', '%', '(', ')', '[', ']' ])
#     tokens = tokenizer.tokenize(text)
#     tokens2 = tokenizer.tokenize(text2)
#     print(tokens)
#     print(tokens2)
