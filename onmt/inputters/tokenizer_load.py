import sentencepiece as spm
from .tokenizer import Korean_tokenizer, English_tokenizer

def Korean_tokenizer_load(x):

    Korean_tokenizer()

    sp = spm.SentencePieceProcessor()
    sp.Load('../../data/korean_tok.model')

    return sp.EncodeAsPieces(x)


def English_tokenizer_load(x):

    English_tokenizer()

    sp = spm.SentencePieceProcessor()
    sp.Load('../../data/english_tok.model')

    return sp.EncodeAsPieces(x)