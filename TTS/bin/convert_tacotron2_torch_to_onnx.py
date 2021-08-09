
from TTS.tts.onnx.models.tacotron2 import Tacotron2
import argparse
import torch

from TTS.tts.utils.synthesis import numpy_to_torch, text_to_seqvec
from TTS.tts.utils.text.symbols import phonemes, symbols, make_symbols
from TTS.utils.io import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--torch_model_path',
                    type=str,
                    help='Path to target torch model to be converted to ONNX.')
parser.add_argument('--config_path',
                    type=str,
                    help='Path to config file of torch model.')
parser.add_argument('--output_path',
                    type=str,
                    help='path to output file including file name to save ONNX model.')
args = parser.parse_args()

# load model config
config_path = args.config_path
c = load_config(config_path)
num_speakers = 0

# if the vocabulary was passed, replace the default
if 'characters' in c.keys():
    symbols, phonemes = make_symbols(**c.characters)

# init torch model
num_chars = len(phonemes) if c.use_phonemes else len(symbols)

with torch.no_grad():
    model = torch.jit.script(Tacotron2(num_chars=num_chars + getattr(c, "add_blank", False),
                                       num_speakers=num_speakers,
                                       r=c.r,
                                       postnet_output_dim=c.audio['num_mels'],
                                       decoder_output_dim=c.audio['num_mels'],
                                       gst=c.use_gst,
                                       gst_embedding_dim=c.gst['gst_embedding_dim'],
                                       gst_num_heads=c.gst['gst_num_heads'],
                                       gst_style_tokens=c.gst['gst_style_tokens'],
                                       gst_use_speaker_embedding=c.gst['gst_use_speaker_embedding'],
                                       attn_type=c.attention_type,
                                       attn_win=c.windowing,
                                       attn_norm=c.attention_norm,
                                       prenet_type=c.prenet_type,
                                       prenet_dropout=c.prenet_dropout,
                                       forward_attn=c.use_forward_attn,
                                       trans_agent=c.transition_agent,
                                       forward_attn_mask=c.forward_attn_mask,
                                       location_attn=c.location_attn,
                                       attn_K=c.attention_heads,
                                       separate_stopnet=c.separate_stopnet,
                                       bidirectional_decoder=c.bidirectional_decoder,
                                       double_decoder_consistency=c.double_decoder_consistency,
                                       ddc_r=c.ddc_r,
                                       speaker_embedding_dim=None))

    checkpoint = torch.load(args.torch_model_path,
                            map_location=torch.device('cpu'))
    state_dict = checkpoint['model']

    new_state_dict = {}
    # remap some changed state keys
    for (key, value) in state_dict.items():
        if key.endswith("weight_ih") or key.endswith("weight_hh") or key.endswith("bias_ih") or key.endswith("bias_hh"):
            new_state_dict[key + "_l0"] = value
        else:
            new_state_dict[key] = value

    # model has some state preinitialized that the normal one doesn't
    model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    dummy_input = "hello world"

    # preprocess the given text
    dummy_input = text_to_seqvec(dummy_input, c)
    dummy_input = numpy_to_torch(dummy_input, torch.long)
    dummy_input = dummy_input.unsqueeze(0)

    out = model(dummy_input)

    torch.onnx.export(
        model,
        dummy_input,
        args.output_path,
        opset_version=11,
        input_names=['input'],
        output_names=['decoder_outputs', 'postnet_outputs', 'alignments', 'stop_tokens'],
        dynamic_axes={
            'input': {1: 'in'},
            'decoder_outputs': {1: 'out'},
            'postnet_outputs': {1: 'out'},
            'alignments': {2: 'in', 1: 'out'},
            'stop_tokens': {1: 'out'},
        },
        example_outputs=out,
    )
