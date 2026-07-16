"""
Forced Alignment with Whisper
C. Max Bain
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Union, List

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.types import (
    AlignedTranscriptionResult,
    SingleSegment,
    SingleAlignedSegment,
    SegmentData,
    CharAlignmentArrays,
)
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

PUNKT_ABBREVIATIONS = ['dr', 'vs', 'mr', 'mrs', 'prof']

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]


DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
}

def load_align_model(language_code: str, device: str, model_name: Optional[str] = None, model_dir=None):
    """Load a wav2vec2 alignment model (torchaudio bundle or HF checkpoint) and its char dictionary."""
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            print(f"There is no default alignment model set for this language ({language_code}).\
                Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir)
        except Exception as e:
            print(e)
            print(f"Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
            raise ValueError(f'The chosen align_model "{model_name}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

    return align_model, align_metadata

def _prepare_audio(audio: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Coerce a path, array, or tensor into a [channels, time] tensor."""
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    return audio


def _prepare_alignment(transcript: Iterable[SingleSegment], model_lang: str, model_dictionary: dict,
                       print_progress: bool, combined_progress: bool, ) -> dict:
    """
    Prepare transcript metadata used during alignment.
    """
    total_segments = len(transcript)
    segment_data: dict[int, SegmentData] = {}
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")
            
        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            # wav2vec2 models use "|" character to represent spaces
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")
            
            # ignore whitespace at beginning and end of transcript
            if cdx < num_leading:
                pass
            elif cdx > len(text) - num_trailing - 1:
                pass
            elif char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any([c in model_dictionary.keys() for c in wrd.lower()]):
                clean_wdx.append(wdx)

        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": clean_wdx,
            "sentence_spans": sentence_spans
        }
    return segment_data

def _compute_emission(
        waveform_segment: torch.Tensor,
        model: torch.nn.Module,
        model_type: str,
        device: str,
):
    """Run one waveform segment through the align model and return log-softmax CTC emissions."""
    if waveform_segment.shape[-1] < 400:
        lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
        waveform_segment = torch.nn.functional.pad(
            waveform_segment, (0, 400 - waveform_segment.shape[-1])
        )
    else:
        lengths = None
    with torch.inference_mode():
        if model_type == "torchaudio":
            emissions, _ = model(waveform_segment.to(device), lengths=lengths)
        elif model_type == "huggingface":
            emissions = model(waveform_segment.to(device)).logits
        else:
            raise NotImplementedError(f"Align model of type {model_type} not supported.")
        emissions = torch.log_softmax(emissions, dim=-1)
    return emissions[0].cpu().detach()

def _compute_emission_batch(
        waveform_segments: list[torch.Tensor],
        model: torch.nn.Module,
        model_type: str,
        device: str,
) -> list[torch.Tensor]:
    """Compute CTC emissions for variable-length mono waveforms."""
    if len(waveform_segments) == 0:
        return []

    flattened_waveforms = []
    for waveform_segment in waveform_segments:
        if waveform_segment.ndim == 2 and waveform_segment.shape[0] == 1:
            waveform_segment = waveform_segment.squeeze(0)
        elif waveform_segment.ndim != 1:
            raise ValueError(
                "Each waveform segment must have shape [time] or [1, time], "
                f"but found {list(waveform_segment.shape)}."
            )
        flattened_waveforms.append(waveform_segment)

    with torch.inference_mode():
        if model_type == "torchaudio":
            # The English wav2vec2 bundle uses GroupNorm across the feature
            # extractor's time axis. Running zero-padded waveforms through it
            # as one batch changes the valid features of shorter items. Only
            # the first convolution block contains GroupNorm, so run that block
            # per item and batch the remaining convolution blocks.
            first_block_sequences = []
            first_block = model.feature_extractor.conv_layers[0]
            for waveform_segment in flattened_waveforms:
                if waveform_segment.shape[-1] < 400:
                    waveform_segment = torch.nn.functional.pad(
                        waveform_segment,
                        (0, 400 - waveform_segment.shape[-1]),
                    )
                features, _ = first_block(
                    waveform_segment.unsqueeze(0).unsqueeze(0).to(device),
                    length=None,
                )
                first_block_sequences.append(
                    features.squeeze(0).transpose(0, 1)
                )

            # pad_sequence uses [time, channels], while the convolution blocks
            # consume [batch, channels, time].
            output_lengths = torch.as_tensor(
                [features.shape[0] for features in first_block_sequences],
                dtype=torch.long,
                device=device,
            )
            features = torch.nn.utils.rnn.pad_sequence(
                first_block_sequences,
                batch_first=True,
            ).transpose(1, 2)

            for conv_layer in model.feature_extractor.conv_layers[1:]:
                features, output_lengths = conv_layer(features, output_lengths)
            features = features.transpose(1, 2)
            emissions = model.encoder(features, lengths=output_lengths)
            if model.aux is not None:
                emissions = model.aux(emissions)
        elif model_type == "huggingface":
            # Padding variable-length inputs before a GroupNorm feature
            # extractor changes the normalization statistics for shorter
            # items. Only use the padded batch path for the LayerNorm
            # architecture we have validated. Preserve alignment support for
            # GroupNorm and unknown/custom architectures by using the existing
            # single-item forward path.
            if getattr(model.config, "feat_extract_norm", None) != "layer":
                return [
                    _compute_emission(
                        waveform_segment.unsqueeze(0),
                        model,
                        model_type,
                        device,
                    )
                    for waveform_segment in flattened_waveforms
                ]
            lengths = torch.as_tensor(
                [waveform_segment.shape[-1] for waveform_segment in flattened_waveforms],
                dtype=torch.long,
                device=device,
            )
            waveform_batch = torch.nn.utils.rnn.pad_sequence(
                flattened_waveforms,
                batch_first=True,
            )
            if waveform_batch.shape[-1] < 400:
                waveform_batch = torch.nn.functional.pad(
                    waveform_batch,
                    (0, 400 - waveform_batch.shape[-1]),
                )
            waveform_batch = waveform_batch.to(device)
            # Hugging Face models need a sample-level mask so padded audio does
            # not participate in transformer attention. Treat the 400-sample
            # minimum padding as valid for very short inputs, matching the
            # existing single-item Hugging Face path.
            effective_lengths = lengths.clamp_min(400)
            attention_mask = (
                torch.arange(waveform_batch.shape[-1], device=device).unsqueeze(0)
                < effective_lengths.unsqueeze(1)
            ).to(torch.long)
            emissions = model(
                waveform_batch,
                attention_mask=attention_mask,
            ).logits
            output_lengths = model._get_feat_extract_output_lengths(effective_lengths)
        else:
            raise NotImplementedError(f"Align model of type {model_type} not supported.")
        emissions = torch.log_softmax(emissions, dim=-1)

    if output_lengths is None:
        raise RuntimeError(
            f"Align model of type {model_type} did not return output lengths."
        )

    emissions = emissions.cpu().detach()
    output_lengths = output_lengths.cpu().tolist()
    return [
        emission[:output_length]
        for emission, output_length in zip(emissions, output_lengths)
    ]


def _char_segments_from_alignment(
    segment_data: SegmentData,
    text: str,
    char_segments: list,
    t1: float,
    ratio: float,
    model_lang: str,
) -> CharAlignmentArrays:
    """Map backtracked char segments onto per-character start/end/score arrays and word ids."""
    # assign timestamps to aligned characters
    chars = list(text)
    starts = np.full(len(chars), np.nan, dtype=np.float64)
    ends = np.full(len(chars), np.nan, dtype=np.float64)
    scores = np.full(len(chars), np.nan, dtype=np.float64)
    word_ids = np.empty(len(chars), dtype=np.int32)

    for cdx, char_segment in zip(
        segment_data["clean_cdx"],
        char_segments,
        strict=True,
    ):
        starts[cdx] = round(char_segment.start * ratio + t1, 3)
        ends[cdx] = round(char_segment.end * ratio + t1, 3)
        scores[cdx] = round(char_segment.score, 3)

    # Preserve WhisperX's existing word grouping exactly. Spaces are assigned
    # to the following word and later excluded from timestamp/score reduction.
    word_idx = 0
    for cdx in range(len(chars)):
        word_ids[cdx] = word_idx
        # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            word_idx += 1
        elif cdx == len(chars) - 1 or chars[cdx + 1] == " ":
            word_idx += 1

    return CharAlignmentArrays(chars, starts, ends, scores, word_ids)


def _nan_min(values: np.ndarray) -> float:
    """Min over non-NaN entries, or NaN if all entries are NaN."""
    present = values[~np.isnan(values)]
    return float(present.min()) if present.size else np.nan


def _nan_max(values: np.ndarray) -> float:
    """Max over non-NaN entries, or NaN if all entries are NaN."""
    present = values[~np.isnan(values)]
    return float(present.max()) if present.size else np.nan


def _nan_mean(values: np.ndarray) -> np.float64:
    """Mean over non-NaN entries, or NaN if all entries are NaN."""
    if np.isnan(values).all():
        return np.float64(np.nan)
    # This matches pandas Series.mean()'s summation and half-way rounding,
    # including when missing characters occur within a word.
    return np.nanmean(values)


def _interpolate_missing(values: list[float], method: str) -> np.ndarray:
    """Fill missing sentence timestamps using their neighboring sentence indices."""
    result = np.asarray(values, dtype=np.float64)
    if method == "ignore":
        return result
    if method not in {"nearest", "linear"}:
        raise ValueError(
            f"interpolate_method must be 'nearest', 'linear', or 'ignore', found {method!r}."
        )

    known = np.flatnonzero(~np.isnan(result))
    if known.size == 0:
        return result
    if known.size == 1:
        result.fill(result[known[0]])
        return result

    missing = np.flatnonzero(np.isnan(result))
    if method == "linear":
        result[missing] = np.interp(missing, known, result[known])
        return result

    # Nearest chooses the earlier sentence at an exactly equidistant index.
    # Clipping also fills values before the first and after the last timestamp.
    right_positions = np.searchsorted(known, missing).clip(max=known.size - 1)
    left_positions = (right_positions - 1).clip(min=0)
    left = known[left_positions]
    right = known[right_positions]
    nearest = np.where(missing - left <= right - missing, left, right)
    result[missing] = result[nearest]
    return result


def _get_sentence_words(
    char_alignments: CharAlignmentArrays,
    start: int,
    stop: int,
) -> list[dict]:
    """Group aligned characters in [start, stop) into word dicts with start/end/score."""
    sentence_words = []
    chars = char_alignments.chars
    word_ids = char_alignments.word_ids
    word_start = start

    while word_start < stop:
        word_id = word_ids[word_start]
        word_stop = word_start + 1
        while word_stop < stop and word_ids[word_stop] == word_id:
            word_stop += 1

        word_text = "".join(chars[word_start:word_stop]).strip()
        if len(word_text) == 0:
            word_start = word_stop
            continue

        # dont use space character for alignment
        aligned_indices = [
            index
            for index in range(word_start, word_stop)
            if chars[index] != " "
        ]
        aligned_start = _nan_min(char_alignments.starts[aligned_indices])
        aligned_end = _nan_max(char_alignments.ends[aligned_indices])
        aligned_score = round(_nan_mean(char_alignments.scores[aligned_indices]), 3)

        # -1 indicates unalignable
        word_segment = {"word": word_text}

        if not np.isnan(aligned_start):
            word_segment["start"] = aligned_start
        if not np.isnan(aligned_end):
            word_segment["end"] = aligned_end
        if not np.isnan(aligned_score):
            word_segment["score"] = aligned_score

        sentence_words.append(word_segment)
        word_start = word_stop
    return sentence_words


def _aligned_subsegments(
    char_alignments: CharAlignmentArrays,
    segment_data: SegmentData,
    text: str,
    return_char_alignments: bool,
    interpolate_method: str,
    model_lang: str,
) -> list[SingleAlignedSegment]:
    """Split a segment's char alignments into sentence-level records, interpolating missing timestamps."""
    aligned_subsegments = []
    for sstart, send in segment_data["sentence_spans"]:
        # Punkt's end offset is exclusive for sentence text. WhisperX's
        # existing character selection also includes the character at `send`,
        # commonly the following space, so preserve that behavior here.
        char_stop = min(send + 1, len(char_alignments.chars))

        sentence_text = text[sstart:send]
        sentence_start = _nan_min(char_alignments.starts[sstart:char_stop])
        non_space_indices = [
            index
            for index in range(sstart, char_stop)
            if char_alignments.chars[index] != " "
        ]
        sentence_end = _nan_max(char_alignments.ends[non_space_indices])

        sentence_words = _get_sentence_words(char_alignments, sstart, char_stop)

        aligned_subsegments.append({
            "text": sentence_text,
            "start": sentence_start,
            "end": sentence_end,
            "words": sentence_words,
        })

        if return_char_alignments:
            sentence_chars = []
            for index in range(sstart, char_stop):
                char = {"char": char_alignments.chars[index]}
                if not np.isnan(char_alignments.starts[index]):
                    char["start"] = float(char_alignments.starts[index])
                if not np.isnan(char_alignments.ends[index]):
                    char["end"] = float(char_alignments.ends[index])
                if not np.isnan(char_alignments.scores[index]):
                    char["score"] = float(char_alignments.scores[index])
                sentence_chars.append(char)
            aligned_subsegments[-1]["chars"] = sentence_chars

    starts = _interpolate_missing(
        [segment["start"] for segment in aligned_subsegments],
        interpolate_method,
    )
    ends = _interpolate_missing(
        [segment["end"] for segment in aligned_subsegments],
        interpolate_method,
    )

    # Concatenate sentences assigned the same timestamps. Sorting keys matches
    # pandas groupby's default output order; NaN keys are omitted as before.
    separator = "" if model_lang in LANGUAGES_WITHOUT_SPACES else " "
    grouped: dict[tuple[float, float], SingleAlignedSegment] = {}
    for segment, start, end in zip(aligned_subsegments, starts, ends, strict=True):
        if np.isnan(start) or np.isnan(end):
            continue
        key = (float(start), float(end))
        if key not in grouped:
            grouped[key] = {
                "start": start,
                "end": end,
                "text": segment["text"],
                "words": list(segment["words"]),
            }
            if return_char_alignments:
                grouped[key]["chars"] = list(segment["chars"])
            continue

        grouped_segment = grouped[key]
        grouped_segment["text"] += separator + segment["text"]
        grouped_segment["words"].extend(segment["words"])
        if return_char_alignments:
            grouped_segment["chars"].extend(segment["chars"])

    return [grouped[key] for key in sorted(grouped)]


def _finish_alignment_segment(
    segment: SingleSegment,
    segment_data: SegmentData,
    waveform_segment: torch.Tensor,
    emission: torch.Tensor,
    model_dictionary: dict,
    model_lang: str,
    interpolate_method: str,
    return_char_alignments: bool,
) -> Optional[list[SingleAlignedSegment]]:
    """Convert one CTC emission matrix into aligned sentence and word records."""
    text = segment["text"]
    text_clean = "".join(segment_data["clean_char"])
    tokens = [model_dictionary[c] for c in text_clean]

    blank_id = 0
    for char, code in model_dictionary.items():
        if char == "[pad]" or char == "<pad>":
            blank_id = code

    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack(trellis, emission, tokens, blank_id)
    if path is None:
        return None

    char_segments = merge_repeats(path, text_clean)
    duration = segment["end"] - segment["start"]
    ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)
    char_alignments = _char_segments_from_alignment(
        segment_data,
        text,
        char_segments,
        segment["start"],
        ratio,
        model_lang,
    )
    return _aligned_subsegments(
        char_alignments,
        segment_data,
        text,
        return_char_alignments,
        interpolate_method,
        model_lang,
    )


def _assemble_alignment_result(
    aligned_segments: list[SingleAlignedSegment],
) -> AlignedTranscriptionResult:
    """Flatten per-segment alignment results into the final segments/word_segments output shape."""
    word_segments = [
        word
        for segment in aligned_segments
        for word in segment["words"]
    ]
    return {"segments": aligned_segments, "word_segments": word_segments}


def _unaligned_segment(
    segment: SingleSegment,
    return_char_alignments: bool,
) -> SingleAlignedSegment:
    """Build a fallback aligned-segment record (no word/char timestamps) for segments that can't be aligned."""
    aligned_segment: SingleAlignedSegment = {
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"],
        "words": [],
        "chars": None,
    }
    if return_char_alignments:
        aligned_segment["chars"] = []
    return aligned_segment


def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    """

    audio = _prepare_audio(audio)

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # 1. Preprocess to keep only characters in dictionary
    segment_data = _prepare_alignment(transcript, model_lang, model_dictionary, print_progress, combined_progress)
            
    aligned_segments: List[SingleAlignedSegment] = []
    
    # 2. Get prediction matrix from alignment model & align
    for sdx, segment in enumerate(transcript):
        
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        aligned_seg = _unaligned_segment(segment, return_char_alignments)

        # check we can align
        if len(segment_data[sdx]["clean_char"]) == 0:
            print(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= MAX_DURATION:
            print(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping...')
            aligned_segments.append(aligned_seg)
            continue

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        waveform_segment = audio[:, f1:f2]
        emission = _compute_emission(waveform_segment, model, model_type, device)
        aligned_subsegments = _finish_alignment_segment(
            segment,
            segment_data[sdx],
            waveform_segment,
            emission,
            model_dictionary,
            model_lang,
            interpolate_method,
            return_char_alignments,
        )
        if aligned_subsegments is None:
            print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
            aligned_segments.append(aligned_seg)
            continue
        aligned_segments += aligned_subsegments

    return _assemble_alignment_result(aligned_segments)


def align_batch(
    transcripts: list[list[SingleSegment]],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: list[Union[str, np.ndarray, torch.Tensor]],
    device: str,
    batch_size: int,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> list[AlignedTranscriptionResult]:
    """Align multiple transcripts/audios together, batching same-sized CTC forward passes across inputs for throughput."""
    if len(transcripts) != len(audio):
        raise ValueError(
            "transcripts and audio must contain the same number of items, "
            f"but found {len(transcripts)} transcripts and {len(audio)} audio items."
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, but found {batch_size}.")

    audios = [_prepare_audio(audio_item) for audio_item in audio]
    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # A transcript segment can expand into several sentence-level output
    # segments, so keep one result slot per original segment while acoustic
    # jobs are flattened across audio inputs.
    segment_results: list[list[Optional[list[SingleAlignedSegment]]]] = [
        [None] * len(transcript)
        for transcript in transcripts
    ]
    segment_jobs = []
    total_segments = sum(len(transcript) for transcript in transcripts)
    processed_segments = 0

    for audio_index, (transcript, audio_item) in enumerate(zip(transcripts, audios)):
        max_duration = audio_item.shape[1] / SAMPLE_RATE
        segment_data = _prepare_alignment(
            transcript,
            model_lang,
            model_dictionary,
            print_progress=False,
            combined_progress=False,
        )

        for segment_index, segment in enumerate(transcript):
            processed_segments += 1
            if print_progress and total_segments:
                base_progress = processed_segments / total_segments * 100
                percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")

            fallback = _unaligned_segment(segment, return_char_alignments)
            if len(segment_data[segment_index]["clean_char"]) == 0:
                print(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original...')
                segment_results[audio_index][segment_index] = [fallback]
                continue

            if segment["start"] >= max_duration:
                print(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping...')
                segment_results[audio_index][segment_index] = [fallback]
                continue

            f1 = int(segment["start"] * SAMPLE_RATE)
            f2 = int(segment["end"] * SAMPLE_RATE)
            segment_jobs.append({
                "audio_index": audio_index,
                "segment_index": segment_index,
                "segment": segment,
                "segment_data": segment_data[segment_index],
                "waveform_segment": audio_item[:, f1:f2],
                "fallback": fallback,
            })

    # Group similarly sized windows to reduce padding in transformer batches.
    # Results are written back through audio/segment indices, so this does not
    # affect public output ordering.
    segment_jobs.sort(key=lambda job: job["waveform_segment"].shape[-1])

    for batch_start in range(0, len(segment_jobs), batch_size):
        batch_jobs = segment_jobs[batch_start:batch_start + batch_size]
        emissions = _compute_emission_batch(
            [job["waveform_segment"] for job in batch_jobs],
            model,
            model_type,
            device,
        )
        if len(emissions) != len(batch_jobs):
            raise RuntimeError(
                "Batch alignment model returned an unexpected number of emissions: "
                f"expected {len(batch_jobs)}, found {len(emissions)}."
            )

        for job, emission in zip(batch_jobs, emissions):
            aligned_subsegments = _finish_alignment_segment(
                job["segment"],
                job["segment_data"],
                job["waveform_segment"],
                emission,
                model_dictionary,
                model_lang,
                interpolate_method,
                return_char_alignments,
            )
            if aligned_subsegments is None:
                print(f'Failed to align segment ("{job["segment"]["text"]}"): backtrack failed, resorting to original...')
                aligned_subsegments = [job["fallback"]]
            segment_results[job["audio_index"]][job["segment_index"]] = aligned_subsegments

    results = []
    for audio_segment_results in segment_results:
        aligned_segments = []
        for aligned_subsegments in audio_segment_results:
            if aligned_subsegments is None:
                raise RuntimeError("Internal error: an alignment segment was not processed.")
            aligned_segments.extend(aligned_subsegments)
        results.append(_assemble_alignment_result(aligned_segments))
    return results

"""
source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
"""


def get_trellis(emission, tokens, blank_id=0):
    """Build the DP trellis of cumulative log-probabilities for aligning `tokens` to CTC `emission` frames."""
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra dimensions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    """Trace the highest-probability path through the trellis; returns None if it fails to reach token 0."""
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else blank_id].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        # failed
        return None

    return path[::-1]


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def merge_repeats(path, transcript):
    """Collapse consecutive path points with the same token index into per-character Segments."""
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    """Join character Segments into word Segments, splitting on the separator label."""
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words
