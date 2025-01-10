import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.nn.functional as F
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

# Global variables for models
spec_generator = None
model = None
device = None

def setup_models():
    global spec_generator, model, device
    
    device = torch_xla.device()
    
    # Initialize models
    spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch").to(device)
    model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan").to(device)
    
    # Convert to FP32 instead of FP16
    model = convert_to_dtype(model, dtype=torch.float32)
    spec_generator = convert_to_dtype(spec_generator, dtype=torch.float32)
    
    # Set eval mode
    spec_generator.eval()
    model.eval()
    
    # Initialize cache
    torch_xla.runtime.initialize_cache('./compilation_cache', readonly=False)

def convert_to_dtype(model, dtype=torch.float16):
    # Convert model parameters to FP16
    model = model.to(dtype)
    
    # Keep norm layers in FP32 (for future models)
    for name, param in model.named_parameters():
        if 'bn' in name:
            param.data = param.data.float()
            param.grad = param.grad.float()
    
    return model

def parse_text(text: str | list[str], padlen: int = 128) -> tuple[torch.Tensor, list[int]]:
    torch.set_default_device('cpu')

    pre_pad_lens = torch.zeros(len(text))
    text = [text] if isinstance(text, str) else text # wrap it in a list if required
    out = torch.zeros(len(text), padlen) # preallocate the output

    for index, t in enumerate(text):
        parsed = spec_generator.parse(t).cpu()
        pre_pad_len = parsed.shape[1]
        
        if pre_pad_len > padlen:
            print(f'WARNING: Input padding is insufficient for text size. Recommend doubling padding length.')
            
        out[index, :] = F.pad(parsed.cpu(), (0, padlen - parsed.shape[1]), value=0).long()
        pre_pad_lens[index] = pre_pad_len

    return out.long().cpu(), pre_pad_lens

def static_regulate_len(
    durations,
    enc_out,
    pace: float = 1.0,
    group_size: int = 1,
    dur_lens: torch.tensor = None,
    max_allowed_len: int = 768,
):
    """XLA-optimized version of regulate_len that minimizes dynamic operations.
    Uses a pre-defined maximum length to avoid dynamic arange operations."""
    dtype = enc_out.dtype
    device = enc_out.device
    
    # Static division and floor operations
    reps = (durations.float() / pace + 0.5).floor()
    dec_lens = reps.sum(dim=1)
    
    # Pre-compute pad and cumsum in a more static way
    padded_reps = torch.nn.functional.pad(reps, (1, 0, 0, 0), value=0.0)
    reps_cumsum = torch.cumsum(padded_reps, dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype=dtype, device=device)
    
    # Use static pre-computed range tensor instead of dynamic arange
    static_range = torch.arange(max_allowed_len, device=device)[None, :, None]
    
    # Compute multiplication mask with fewer dynamic operations
    mult = (reps_cumsum[:, :, :-1] <= static_range) & (reps_cumsum[:, :, 1:] > static_range)
    mult = mult.to(dtype)
    
    # Final matrix multiplication
    enc_rep = torch.matmul(mult, enc_out)
    
    # Trim to actual length needed
    length_mask = torch.arange(enc_rep.shape[1], device=enc_rep.device)[None, :] < dec_lens[:, None]
    enc_rep = enc_rep * length_mask.unsqueeze(-1)

    return enc_rep, dec_lens, dec_lens

@torch.compile(backend='openxla', fullgraph=True)
def fp_prediction(parsed: torch.Tensor, conditioning=None):
    with torch.no_grad():
        parsed = parsed.to(device)     
        enc_out, enc_mask = spec_generator.fastpitch.encoder(input=parsed, conditioning=conditioning)
        # Remove the FP16 conversions
        enc_out, enc_mask = enc_out, enc_mask
        
        # Duration prediction
        log_durs_predicted = spec_generator.fastpitch.duration_predictor(enc_out, enc_mask, conditioning=conditioning)
        durs_predicted = log_to_duration(
            log_dur=log_durs_predicted, 
            min_dur=spec_generator.fastpitch.min_token_duration, 
            max_dur=spec_generator.fastpitch.max_token_duration, 
            mask=enc_mask
        )
        
        # Pitch prediction
        pitch_predicted = spec_generator.fastpitch.pitch_predictor(enc_out, enc_mask, conditioning=conditioning)
        pitch_emb = spec_generator.fastpitch.pitch_emb(pitch_predicted.unsqueeze(1))
        
        # Combine encoder output and pitch embedding
        enc_out = enc_out + pitch_emb.transpose(1, 2)
        
        len_regulated, dec_lens, real_len = static_regulate_len(durs_predicted, enc_out, pace=1.0)
        dec_out, _ = spec_generator.fastpitch.decoder(input=len_regulated, seq_lens=dec_lens, conditioning=conditioning)
        dec_out = dec_out
        
        spect = spec_generator.fastpitch.proj(dec_out).transpose(1, 2) # Obtain spectrogram
        audio = model.convert_spectrogram_to_audio(spec=spect)
        audio, durs_predicted = audio.to('cpu'), durs_predicted.to('cpu')

        return audio, real_len, durs_predicted

def infer_e2e(text: list[str], parsed: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    '''
    This function can optionally consume a directly parsed tensor
    for debugging purposes.
    '''
    pre_pad_len: int | None = None
    
    if parsed is None:
        assert type(text) is list, f'Invalid input type. Got: {type(text)} expected a list'
        parsed, pre_pad_len = parse_text(text)

    audio, real_len, durs_predicted = fp_prediction(parsed.to(device))
    
    return audio, pre_pad_len, real_len, durs_predicted

@torch.compile(backend="openxla", fullgraph=True)
def log_to_duration(log_dur, min_dur, max_dur, mask):
    dur = torch.clamp(torch.exp(log_dur) - 1.0, min_dur, max_dur)
    dur *= mask.squeeze(2)
    return dur

def tensor_to_int(x: torch.Tensor) -> int:
    '''
    Casts tensors to integer and places them on CPU 
    '''
    return x.cpu().detach().long().item()