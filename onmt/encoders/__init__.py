"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
# from onmt.encoders.ggnn_encoder import GGNNEncoder
# from onmt.encoders.rnn_encoder import RNNEncoder
# from onmt.encoders.cnn_encoder import CNNEncoder
# from onmt.encoders.mean_encoder import MeanEncoder
# from onmt.encoders.audio_encoder import AudioEncoder
# from onmt.encoders.image_encoder import ImageEncoder


str2enc = {"transformer": TransformerEncoder}

__all__ = ["EncoderBase", "TransformerEncoder" "str2enc"]
