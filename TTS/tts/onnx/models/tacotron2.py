from torch import nn

from TTS.tts.layers.tacotron2 import Postnet
from TTS.tts.onnx.layers.tacotron2 import Encoder, Decoder
from TTS.tts.models.tacotron_abstract import TacotronAbstract

class Tacotron2(TacotronAbstract):
    def __init__(self,
                 num_chars,
                 num_speakers,
                 r,
                 postnet_output_dim=80,
                 decoder_output_dim=80,
                 attn_type='original',
                 attn_win=False,
                 attn_norm="softmax",
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 attn_K=5,
                 separate_stopnet=True,
                 bidirectional_decoder=False,
                 double_decoder_consistency=False,
                 ddc_r=None,
                 encoder_in_features=512,
                 decoder_in_features=512,
                 speaker_embedding_dim=None,
                 gst=False,
                 gst_embedding_dim=512,
                 gst_num_heads=4,
                 gst_style_tokens=10,
                 gst_use_speaker_embedding=False):
        super(Tacotron2,
              self).__init__(num_chars, num_speakers, r, postnet_output_dim,
                             decoder_output_dim, attn_type, attn_win,
                             attn_norm, prenet_type, prenet_dropout,
                             forward_attn, trans_agent, forward_attn_mask,
                             location_attn, attn_K, separate_stopnet,
                             bidirectional_decoder, double_decoder_consistency,
                             ddc_r, encoder_in_features, decoder_in_features,
                             speaker_embedding_dim, gst, gst_embedding_dim,
                             gst_num_heads, gst_style_tokens, gst_use_speaker_embedding)

        assert self.num_speakers <= 1
        # speaker embedding layer
        # if self.num_speakers > 1:
        #     if not self.embeddings_per_sample:
        #         speaker_embedding_dim = 512
        #         self.speaker_embedding = nn.Embedding(self.num_speakers, speaker_embedding_dim)
        #         self.speaker_embedding.weight.data.normal_(0, 0.3)

        # speaker and gst embeddings is concat in decoder input
        # if self.num_speakers > 1:
        #     self.decoder_in_features += speaker_embedding_dim # add speaker embedding dim

        # embedding layer
        self.embedding = nn.Embedding(num_chars, 512, padding_idx=0)

        # base model layers
        self.encoder = Encoder(self.encoder_in_features)
        self.decoder = Decoder(self.decoder_in_features, self.decoder_output_dim, r, attn_type, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, attn_K, separate_stopnet)
        self.postnet = Postnet(self.postnet_output_dim)

        assert not self.gst
        # global style token layers
        # if self.gst:
        #     self.gst_layer = GST(num_mel=80,
        #                          num_heads=self.gst_num_heads,
        #                          num_style_tokens=self.gst_style_tokens,
        #                          gst_embedding_dim=self.gst_embedding_dim,
        #                          speaker_embedding_dim=speaker_embedding_dim if self.embeddings_per_sample and self.gst_use_speaker_embedding else None)
        # backward pass decoder
        # if self.bidirectional_decoder:
        #     self._init_backward_decoder()
        # setup DDC
        # if self.double_decoder_consistency:
        #     self.coarse_decoder = Decoder(
        #         self.decoder_in_features, self.decoder_output_dim, ddc_r, attn_type,
        #         attn_win, attn_norm, prenet_type, prenet_dropout, forward_attn,
        #         trans_agent, forward_attn_mask, location_attn, attn_K,
        #         separate_stopnet)

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(self, text):
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs)

        # if self.gst:
        #     # B x gst_dim
        #     encoder_outputs = self.compute_gst(encoder_outputs,
        #                                        style_mel,
        #                                        speaker_embeddings if self.gst_use_speaker_embedding else None)
        # if self.num_speakers > 1:
        #     if not self.embeddings_per_sample:
        #         speaker_embeddings = self.speaker_embedding(speaker_ids)[:, None]
        #     encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaker_embeddings)

        decoder_outputs, alignments, stop_tokens = self.decoder(encoder_outputs)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(
            decoder_outputs, postnet_outputs, alignments)
        return decoder_outputs, postnet_outputs, alignments, stop_tokens

    def inference(self, text):
        self.forward(text)
