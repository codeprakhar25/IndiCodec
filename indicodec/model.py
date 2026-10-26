"""IndiCodec inference graph. No weight download.

Architecture classes still come from installed `neucodec` (Apache 2.0, commit ed3e6cd).
The release model.safetensors holds every tensor, including the 16 w2v-BERT 2.0 (MIT) layers we use, built from the
shipped w2v_bert/config.json. The internal .pt has no w2v-BERT, so that path still fetches it (pinned revision).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEM_LAYER = 16
HOP16, HOP24 = 320, 480
LEVELS_S = [3] * 8
W2V = "facebook/w2v-bert-2.0"
W2V_REV = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
W2V_DIR = Path(__file__).resolve().parent / "w2v_bert"   # config (16 layers) + preprocessor config, from W2V_REV
NEED_GEN = (
    "fsq_s.", "fsq_a.", "backbone.", "head.",
    "CodecEnc.", "SemanticEncoder_module.",
)
# training-only head (a semantic reconstruction loss); never runs at inference, dropped from the release file
TRAIN_ONLY = ("SemanticDecoder_module.",)


class IndiNet(nn.Module):
    """IndiCodec encode / decode only. Refuses phase2d, rope_fix, continuous, joint FSQ."""

    def __init__(self, levels_a: list[int], w2v_bundled: bool = True):
        super().__init__()
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
        from vector_quantize_pytorch import FSQ
        from neucodec.codec_encoder import CodecEncoder
        from neucodec.codec_decoder_vocos import ISTFTHead, VocosBackbone
        from neucodec.module import SemanticEncoder

        self.w2v_bundled = bool(w2v_bundled)
        if self.w2v_bundled:   # weights come from the release file in load_gen
            sm = Wav2Vec2BertModel(Wav2Vec2BertConfig.from_json_file(str(W2V_DIR / "config.json")))
        else:
            sm = Wav2Vec2BertModel.from_pretrained(W2V, revision=W2V_REV, output_hidden_states=True)
            sm.encoder.layers = sm.encoder.layers[:SEM_LAYER]
        if len(sm.encoder.layers) != SEM_LAYER:
            raise SystemExit(f"w2v-BERT has {len(sm.encoder.layers)} layers, IndiCodec needs {SEM_LAYER}")
        self.semantic_model = sm.eval().requires_grad_(False)
        self.SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024)
        self.CodecEnc = CodecEncoder()
        self.backbone = VocosBackbone(hidden_dim=1024, depth=12, heads=16, pos_meb_dim=64)
        self.head = ISTFTHead(dim=1024, n_fft=HOP24 * 4, hop_length=HOP24, padding="same")
        self.fsq_s = FSQ(levels=LEVELS_S, dim=1024)
        self.levels_a = list(levels_a)
        self.fsq_a = FSQ(levels=self.levels_a, dim=1024)
        self.tail_frames = 0

    def load_gen(self, gen: dict) -> None:
        gen = {k: v for k, v in gen.items() if not k.startswith(TRAIN_ONLY)}
        missing_pref = [p for p in NEED_GEN if not any(k.startswith(p) for k in gen)]
        if missing_pref:
            raise SystemExit(f"ckpt gen missing {missing_pref}; this infer needs the IndiCodec trained towers")
        r = self.load_state_dict(gen, strict=False)
        bad_u = [k for k in r.unexpected_keys if k.startswith("head.refine.")]
        if bad_u:
            raise RuntimeError(f"checkpoint has PhaseRefiner2D ({bad_u[0]}); IndiCodec infer refuses it")
        bad_m = [m for m in r.missing_keys if self.w2v_bundled or not m.startswith("semantic_model.")]
        if bad_m:
            raise SystemExit(f"ckpt missing {len(bad_m)} keys, e.g. {bad_m[:5]}")

    @torch.no_grad()
    def semantic_target(self, feats: torch.Tensor) -> torch.Tensor:
        return self.semantic_model(feats).hidden_states[SEM_LAYER].transpose(1, 2)

    def towers(self, wav16: torch.Tensor, feats: torch.Tensor):
        return self.semantic_target(feats), self.CodecEnc(wav16).transpose(1, 2)

    def encode(self, wav16: torch.Tensor, feats: torch.Tensor):
        sem, ac = self.towers(wav16, feats)
        s_enc = self.SemanticEncoder_module(sem)
        t = min(sem.shape[-1], ac.shape[-1], s_enc.shape[-1])
        t -= t % 2
        sem, ac, s_enc = sem[..., :t], ac[..., :t], s_enc[..., :t]
        s25 = F.avg_pool1d(s_enc, 2, 2)
        zq_s, idx_s = self.fsq_s(s25.transpose(1, 2))
        c50 = zq_s.repeat_interleave(2, dim=1)
        res = ac.transpose(1, 2) - c50.detach()
        zq_a, idx_a = self.fsq_a(res)
        return {"idx_s": idx_s, "idx_a": idx_a, "t": t}

    def codes_to_latent(self, idx_s: torch.Tensor, idx_a: torch.Tensor | None) -> torch.Tensor:
        c50 = self.fsq_s.indices_to_codes(idx_s).repeat_interleave(2, dim=1)
        if idx_a is None:
            return c50
        za = self.fsq_a.indices_to_codes(idx_a)
        t = min(c50.shape[1], za.shape[1])
        return c50[:, :t] + za[:, :t]

    def decode_latent(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(h))[0]

    @torch.no_grad()
    def decode_codes(self, idx_s, idx_a, mode: str = "full") -> torch.Tensor:
        if mode == "zero_c":
            return self.decode_latent(self.fsq_a.indices_to_codes(idx_a))
        if mode == "zero_ac":
            return self.decode_latent(self.codes_to_latent(idx_s, None))
        return self.decode_latent(self.codes_to_latent(idx_s, idx_a))


_FE = None


def features(wav16_np_list):
    global _FE
    if _FE is None:
        from transformers import AutoFeatureExtractor
        _FE = AutoFeatureExtractor.from_pretrained(str(W2V_DIR))   # same file as W2V at W2V_REV
    return _FE(wav16_np_list, sampling_rate=16_000, return_tensors="pt").input_features


def to_rate(w, src_sr, tgt):
    from scipy.signal import resample_poly
    if int(src_sr) == int(tgt):
        return np.ascontiguousarray(w, dtype=np.float32)
    g = math.gcd(int(src_sr), int(tgt))
    return np.ascontiguousarray(resample_poly(np.asarray(w, np.float64), tgt // g, src_sr // g), dtype=np.float32)


def pad_w16(w16: np.ndarray, tail_frames: int) -> np.ndarray:
    return np.pad(w16, (0, (-len(w16)) % HOP16 + int(tail_frames) * HOP16))


@torch.no_grad()
def encode_wav(model: IndiNet, wav, sr, device, amp: bool = True):
    w16 = pad_w16(to_rate(wav, sr, 16000), getattr(model, "tail_frames", 0))
    feats = features([w16]).to(device)
    x = torch.from_numpy(w16)[None, None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device != "cpu"):
        e = model.encode(x, feats)
    return e["idx_s"], e["idx_a"], e["t"]
