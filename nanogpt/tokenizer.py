from enum import Enum

import numpy as np

import tokenmonster
import tiktoken

class TokenizerType(Enum):
    CHAR = 'CHAR',
    TOKENMONSTER = 'TOKENMONSTER',
    TIKTOKEN = "TIKTOKEN"

class Tokenizer():
    def __init__(self, type=TokenizerType.CHAR, desired_vocab_size=0):
        if not isinstance(type, TokenizerType):
            raise ValueError("Invalid Tokenizer Type!")
        
        if type == TokenizerType.CHAR:
            chars = " !@#$%^&*()_+1234567890-=qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM[];',./{}:\"<>?`~\n\\"
            self._stoi = { ch:i for i,ch in enumerate(chars)}
            self._itos = { i:ch for i,ch in enumerate(chars)}

            self.vocab_size = len(chars)

            self._encode = lambda s: np.array([self._stoi[c] for c in s]).astype(int)
            self._decode = lambda l: ''.join([self._itos[i] for i in l])

        elif type == TokenizerType.TOKENMONSTER:
            vocab_sizes = [1024, 2048, 4096, 16000, 24000, 32000, 40000, 50256, 65636, 100256]
            vocab_size = 1024
            if desired_vocab_size not in vocab_sizes:
                for vs in vocab_sizes:
                    if vs > vocab_size and vs <= desired_vocab_size:
                        vocab_size = vs
                    elif vs > desired_vocab_size:
                        break
            
            self.vocab_size = vocab_size
            
            self._vocab = tokenmonster.load(f"englishcode-{vocab_size}-clean-v1")
            self._decoder = self._vocab.decoder()
            self._encode = self._vocab.tokenize
            self._decode = self._decoder.decode

        elif type == TokenizerType.TIKTOKEN:
            self.vocab_size = 100256

            self._encoding = tiktoken.get_encoding("cl100k_base")
            self._encode = self._encoding.encode
            self._decode = self._encoding.decode

    def encode(self, text):
        return np.array(self._encode(text)).astype(int)

    def decode(self, tokens):
        return self._decode(tokens)