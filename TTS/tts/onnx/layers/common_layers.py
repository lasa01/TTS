from TTS.tts.layers.attentions import Linear, LocationLayer
from typing_extensions import Final
import torch
from torch import nn
from torch.nn import functional as F


class OriginalAttention(nn.Module):
    """Following the methods proposed here:
        - https://arxiv.org/abs/1712.05884
        - https://arxiv.org/abs/1807.06736 + state masking at inference
        - Using sigmoid instead of softmax normalization
        - Attention windowing at inference time
    """

    windowing: Final[bool]
    norm: Final[str]
    forward_attn: Final[bool]
    trans_agent: Final[bool]
    forward_attn_mask: Final[bool]
    location_attention: Final[bool]

    # Pylint gets confused by PyTorch conventions here
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, query_dim, embedding_dim, attention_dim,
                 location_attention, attention_location_n_filters,
                 attention_location_kernel_size, windowing, norm, forward_attn,
                 trans_agent, forward_attn_mask):
        super(OriginalAttention, self).__init__()
        self.query_layer = Linear(
            query_dim, attention_dim, bias=False, init_gain='tanh')
        self.inputs_layer = Linear(
            embedding_dim, attention_dim, bias=False, init_gain='tanh')
        self.v = Linear(attention_dim, 1, bias=True)
        if trans_agent:
            self.ta = nn.Linear(
                query_dim + embedding_dim, 1, bias=True)
        if location_attention:
            self.location_layer = LocationLayer(
                attention_dim,
                attention_location_n_filters,
                attention_location_kernel_size,
            )
        # self._mask_value = -float("inf")
        self.windowing = windowing
        self.norm = norm
        self.forward_attn = forward_attn
        self.trans_agent = trans_agent
        self.forward_attn_mask = forward_attn_mask
        self.location_attention = location_attention

        # torchscript needs to know all attributes we use
        self.init_states(torch.zeros(1, 1, embedding_dim))


    def init_win_idx(self):
        self.win_idx = -1
        self.win_back = 2
        self.win_front = 6

    def init_forward_attn(self, inputs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.alpha = torch.cat(
            [torch.ones([B, 1]),
             torch.zeros([B, T])[:, :-1] + 1e-7], dim=1).to(inputs.device)
        self.u = (0.5 * torch.ones([B, 1])).to(inputs.device)

    def init_location_attention(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights_cum = torch.zeros([B, T], device=inputs.device)

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights = torch.zeros([B, T], device=inputs.device)
        if self.location_attention:
            self.init_location_attention(inputs)
        if self.forward_attn:
            self.init_forward_attn(inputs)
        if self.windowing:
            self.init_win_idx()

    def preprocess_inputs(self, inputs):
        return self.inputs_layer(inputs)

    def update_location_attention(self, alignments):
        self.attention_weights_cum += alignments

    def get_location_attention(self, query, processed_inputs):
        attention_cat = torch.cat((self.attention_weights.unsqueeze(1),
                                   self.attention_weights_cum.unsqueeze(1)),
                                  dim=1)
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_cat)
        energies = self.v(
            torch.tanh(processed_query + processed_attention_weights +
                       processed_inputs))
        energies = energies.squeeze(-1)
        return energies, processed_query

    def get_attention(self, query, processed_inputs):
        processed_query = self.query_layer(query.unsqueeze(1))
        energies = self.v(torch.tanh(processed_query + processed_inputs))
        energies = energies.squeeze(-1)
        return energies, processed_query

    def apply_windowing(self, attention, inputs):
        back_win = self.win_idx - self.win_back
        front_win = self.win_idx + self.win_front
        if back_win > 0:
            attention[:, :back_win] = -float("inf")
        if front_win < inputs.shape[1]:
            attention[:, front_win:] = -float("inf")
        # this is a trick to solve a special problem.
        # but it does not hurt.
        if self.win_idx == -1:
            attention[:, 0] = attention.max()
        # Update the window
        self.win_idx = torch.argmax(attention, 1).long()[0].item()
        return attention

    def apply_forward_attention(self, alignment):
        # forward attention
        fwd_shifted_alpha = F.pad(
            self.alpha[:, :-1].clone().to(alignment.device), (1, 0, 0, 0))
        # compute transition potentials
        alpha = ((1 - self.u) * self.alpha
                 + self.u * fwd_shifted_alpha
                 + 1e-8) * alignment
        # force incremental alignment
        if not self.training and self.forward_attn_mask:
            _, n = fwd_shifted_alpha.max(1)
            val, _ = alpha.max(1)
            for b in range(alignment.shape[0]):
                alpha[b, n[b] + 3:] = 0
                alpha[b, :(
                    n[b] - 1
                )] = 0  # ignore all previous states to prevent repetition.
                alpha[b,
                      (n[b] - 2
                       )] = 0.01 * val[b]  # smoothing factor for the prev step
        # renormalize attention weights
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha

    def forward(self, query, inputs, processed_inputs):
        """
        shapes:
            query: B x D_attn_rnn
            inputs: B x T_en x D_en
            processed_inputs:: B x T_en x D_attn
            mask: B x T_en
        """
        if self.location_attention:
            attention, _ = self.get_location_attention(
                query, processed_inputs)
        else:
            attention, _ = self.get_attention(
                query, processed_inputs)
        # apply masking
        # if mask is not None:
        #     attention.data.masked_fill_(~mask, self._mask_value)
        # apply windowing - only in eval mode
        if not self.training and self.windowing:
            attention = self.apply_windowing(attention, inputs)

        # normalize attention values
        if self.norm == "softmax":
            alignment = torch.softmax(attention, dim=-1)
        elif self.norm == "sigmoid":
            alignment = torch.sigmoid(attention) / torch.sigmoid(
                attention).sum(
                    dim=1, keepdim=True)
        else:
            raise ValueError("Unknown value for attention norm type")

        if self.location_attention:
            self.update_location_attention(alignment)

        # apply forward attention if enabled
        if self.forward_attn:
            alignment = self.apply_forward_attention(alignment)
            self.alpha = alignment

        context = torch.bmm(alignment.unsqueeze(1), inputs)
        context = context.squeeze(1)
        self.attention_weights = alignment

        # compute transition agent
        if self.forward_attn and self.trans_agent:
            ta_input = torch.cat([context, query.squeeze(1)], dim=-1)
            self.u = torch.sigmoid(self.ta(ta_input))
        return context


def init_attn(attn_type, query_dim, embedding_dim, attention_dim,
              location_attention, attention_location_n_filters,
              attention_location_kernel_size, windowing, norm, forward_attn,
              trans_agent, forward_attn_mask, attn_K):
    if attn_type == "original":
        return OriginalAttention(query_dim, embedding_dim, attention_dim,
                                 location_attention,
                                 attention_location_n_filters,
                                 attention_location_kernel_size, windowing,
                                 norm, forward_attn, trans_agent,
                                 forward_attn_mask)
    raise RuntimeError(
        " [!] Given Attention Type '{attn_type}' is not exist / supported.")
