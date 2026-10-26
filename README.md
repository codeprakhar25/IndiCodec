# IndiCodec

[Audio samples](https://codeprakhar25.github.io/IndiCodec/) · [Weights on Hugging Face](https://huggingface.co/prakharkhatri/IndiCodec)

A 1.09 kbps neural speech codec for building speech language models and TTS. It turns speech into two streams of
integer tokens and decodes them back to 24 kHz audio.

It was trained mostly on English and 10 Indian languages, and it is strongest on Indian languages: among codecs at
about 1 kbps, it keeps the most words in every one of the 10 we tested.

## Key features

- **Low bitrate:** 1.09 kbps, 75 tokens per second.
- **Two token streams:** a content stream (25 Hz) that carries the words, and an acoustic stream (50 Hz) that carries
  the voice and detail.
- **Split FSQ, not RVQ:** each stream is a single FSQ grid. Content is coded first, and one acoustic grid codes what
  remains, instead of a stack of residual codebooks. Each frame is one id per stream, with no learned codebook.
- **16 kHz in, 24 kHz out:** any input sample rate is resampled to 16 kHz.
- **One file:** all weights are in `model.safetensors`. Nothing else is downloaded.
- **Commercial use allowed:** Apache 2.0. All training data is under open licences, none of them non-commercial.

## Model details

| | |
|---|---|
| Content stream | 25 Hz, FSQ 8 × 3 levels = 6,561 ids, 317 bps |
| Acoustic stream | 50 Hz, FSQ 6 × 6 levels = 46,656 ids, 775 bps |
| Bitrate | 1.09 kbps |
| Parameters | 624 M in the file, 387 M of them w2v-BERT 2.0; decoding alone uses 186 M |
| Licence | Apache 2.0 |

The content stream quantises w2v-BERT 2.0 features (layer 16). The acoustic stream quantises what a CNN encoder hears
beyond the content stream. A transformer decoder with an iSTFT head turns the sum back into audio.

IndiCodec is built on [NeuCodec](https://huggingface.co/neuphonic/neucodec) by Neuphonic: it uses NeuCodec's
architecture, and training started from its released weights. The two-stream quantiser, the encoder training and the
training recipe are new.

## Quality

Speech codecs at about 1 kbps, on speakers not seen in training. Every codec is scored on the same clips.

PESQ-WB (higher is better):

| Test set | IndiCodec<br>1.09 kbps | Mimi, 8 codebooks<br>1.10 kbps | SNAC 24 kHz<br>0.98 kbps | NeuCodec<br>0.80 kbps |
|---|---|---|---|---|
| LibriTTS-R test (English) | **2.41** | 2.32 | 2.29 | 2.19 |
| CMU-Arctic (English) | **2.30** | 2.28 | 2.17 | 2.06 |
| Seed-TTS eval (English) | 1.99 | **2.07** | 1.82 | 1.81 |
| FLEURS, 10 Indian languages | **1.90** | 1.84 | 1.69 | 1.69 |

Words after a round trip (lower is better):

| | IndiCodec | Mimi | SNAC | NeuCodec |
|---|---|---|---|---|
| English WER, Seed-TTS eval | **2.5 %** | 3.6 % | 4.9 % | 3.5 % |
| English WER, CMU-Arctic | **2.3 %** | 3.2 % | 3.8 % | 2.8 % |
| Indian-language CER, FLEURS, 10 languages | **6.3 %** | 14.3 % | 13.3 % | 8.9 % |
| Tokens per second | 75 | 100 | 82 | 50 |

English WER uses Whisper large-v3. Indian-language CER uses the IndicConformer 600M ASR; the original recordings score
5.2 %, so IndiCodec adds 1.0 point. The 10 languages are Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia,
Punjabi, Tamil and Telugu.

## Use

```bash
git clone https://github.com/codeprakhar25/IndiCodec && cd IndiCodec
hf download prakharkhatri/IndiCodec model.safetensors --local-dir .
pip install torch==2.6.0 torchaudio==2.6.0 safetensors soundfile transformers==4.51.* vector-quantize-pytorch==1.17.8 torchtune==0.5.0 torchao==0.9.0 local-attention einops librosa
pip install --no-deps "git+https://github.com/neuphonic/neucodec@ed3e6cd"
```

```python
import soundfile as sf
from indicodec import IndiCodec

codec = IndiCodec.load("model.safetensors")
wav, sr = sf.read("speech.wav")

content, acoustic = codec.encode(wav, sr)    # token arrays
audio = codec.decode(content, acoustic)      # 24 kHz
```

## Training data

English: Emilia-YODAS (English part only), LibriTTS, LibriTTS-R and VCTK. Indian languages: Rasa, IndicVoices-R and
IndicVoices from AI4Bharat. Plus about 15 h of laughs, sighs and coughs from VocalSound. All CC BY 4.0, except
VocalSound (CC BY-SA 4.0). No audio is included in this release.

## Limits

- Speech only; not for music.
- Not streaming: the encoder and the decoder both attend over the whole clip.

## Licence

Apache 2.0. Built on [NeuCodec](https://huggingface.co/neuphonic/neucodec) (Apache 2.0). The bundled w2v-BERT 2.0
layers are Meta's, under MIT; its licence text is in `indicodec/w2v_bert/LICENSE`.
