import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List


# Simpler version of autocorrelation
def autocorrelation(query_states, key_states):
    """
    Computes autocorrelation(Q, K) using torch.fft
    
    States are required to be of the same shape of [batch, time_length, embed_dim]
    """

    # rfft: input -> real
    # ifft: input -> complex number
    query_states_fft = torch.fft.rfft(query_states, dim=1)
    key_states_fft = torch.fft.rfft(key_states, dim=1)

    attn_weights = query_states_fft * torch.conj(key_states_fft)
    attn_weights = torch.fft.irfft(attn_weights, dim=1)

    return attn_weights # attn_weights are autocorrelations here.


class AutoformerAttention(nn.Module):
    def __init__(
            self, embed_dim: int, num_heads: int, drop_out: float = 0.0,
            is_decoder: bool = False, bias: bool = True, autocorrelation_factor: int = 3
    ):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = drop_out
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.autocorrelation_factor = autocorrelation_factor
        
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def time_delay_aggregation(self, attn_weights, value_states, autocorrelation_factor=2):
        """
        Computes aggregation as value_states.roll(delay) * top_k_autocorrelation(delay)
        The final result is the autocorrelation-attention output.
        

        The autocorrelation_factor is used to find top-o autocorrelation delays.
        """

        bsz = attn_weights.size(0) // self.num_heads
        num_heads = self.num_heads, 
        tgt_len, channel = attn_weights.size(1), attn_weights.size(2)


        time_length = value_states.size(1)

        autocorrelations = attn_weights.view(bsz, num_heads, tgt_len, channel)

        # find top k autocorrelation delays
        top_k = int(autocorrelation_factor * math.log(time_length))
        autocorrelations_mean_on_head = torch.mean(autocorrelations, dim=(1, -1)) # bsz x tgt_len

        if self.training:
            autocorrelations_mean_on_bsz = torch.mean(autocorrelations_mean_on_head, dim=0)
            _, top_k_delays_index = torch.topk(autocorrelations_mean_on_bsz, top_k)
            top_k_autocorrelation = torch.stack(
                [autocorrelations_mean_on_head[:, top_k_delays_index[i]] for i in range(top_k)], dim = -1
            )
        else:
            top_k_autocorrelation, top_k_delays = torch.topk(autocorrelations_mean_on_head, top_k, dim=1)

        # apply softmax on the channel dim
        top_k_autocrrelations = torch.softmax(top_k_autocorrelation, dim=1) # bsz x top_k

        # compute aggregation: value_states.roll(delay) * top_k_autocorrelations(delay)
        if not self.training:
            # used for computing values_state.roll(delay) in inference
            tmp_values = value_states.repeat(1, 2, 1)
            init_index = (
                torch.arange(time_length)
                .view(1, -1, 1)
                .repeat(bsz * self.num_heads, 1, channel)
                .to(value_states.device)
            )

        delays_agg = torch.zeros_like(value_states).float() # bsz x time_length x channel
        for i in range(top_k):
            # compute value states roll delay
            if not self.training: # inference
                tmp_delay = init_index + top_k_delays_index[:, i].view(-1, 1, 1).repeat(
                    self.num_heads, tgt_len, channel
                )

                value_states_roll_delay = torch.gather(tmp_values, dim=1, index=tmp_delay)
            else:
                value_states_roll_delay = value_states.roll(shifts = -int(top_k_delays[i], dim=1))

            top_k_at_delay = top_k_autocorrelation[:, i]
            # aggregation
            top_k_resized = top_k_at_delay.view(-1, 1, 1).repeat(num_heads, tgt_len, channel)
            delays_agg += value_states_roll_delay * top_k_resized


        attn_output = delays_agg.contiguous()
        return attn_output


    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        
        # (1) period-based dependencies discovery
        # Resize (truncation or zero filling)
        queries_time_length = query_states.size(1)
        values_time_length = value_states.size(1)
        if queries_time_length > values_time_length:
            query_states = query_states[:, : (queries_time_length - values_time_length), :]
            zeros = torch.zeros_like(query_states).float()
            value_states = torch.cat([value_states, zeros], dim=1)
            key_states = torch.cat([key_states, zeros], dim=1)
        else:
            value_states = value_states[:, :queries_time_length, :]
            key_states = key_states[:, :queries_time_length, :]


        # Autocorrelations
        attn_weights = autocorrelation(query_states, key_states)
        
        src_len = key_states.size(1)
        channel = key_states.size(2)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, channel):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, channel)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, channel)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, channel)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, channel)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, channel)
        else:
            attn_weights_reshaped = None
        

        # time delay aggregation
        attn_output = self.time_delay_aggregation(attn_weights, value_states, self.autocorrelation_factor)


        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value