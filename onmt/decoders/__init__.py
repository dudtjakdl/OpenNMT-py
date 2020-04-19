"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
# from onmt.decoders.cnn_decoder import CNNDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder, "transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder",
           "InputFeedRNNDecoder", "str2dec"]
