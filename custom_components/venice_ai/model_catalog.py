"""Decide whether a configured model is still offered by Venice.

Home Assistant is not imported here so the repair rules can be tested
without a running core.
"""

from __future__ import annotations

from dataclasses import dataclass

# POST /audio/speech rejects input longer than this, an unknown format, and a
# voice that does not belong to the selected model. Each of those is HTTP 400.
SPEECH_INPUT_LIMIT = 4096
SPEECH_SPEED_MIN = 0.25
SPEECH_SPEED_MAX = 4.0
SPEECH_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})
_FORMAT_ALIASES = {
    "ogg": "opus",
    "oga": "opus",
    "mpeg": "mp3",
    "wave": "wav",
    "x-wav": "wav",
}
_CONTAINER_PREFERENCE = ("mp3", "wav", "opus", "flac", "aac", "pcm")


def model_ids(models: object, model_type: str | None = None) -> set[str]:
    """Return model ids, optionally limited to one ``model_type`` tag.

    TTS and ASR catalogs are stored in the same list. Filtering by the tag
    the coordinator writes stops an ASR-only refresh from making a TTS model
    look missing.
    """
    if not isinstance(models, list):
        return set()
    ids: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        if model_type is not None and model.get("model_type") != model_type:
            continue
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id:
            ids.add(model_id)
    return ids


def catalog_fetched(
    data: object,
    flag: str,
    models: object,
    model_type: str | None = None,
) -> bool:
    """Return True when this catalog fetch succeeded.

    An explicit flag wins. Older coordinator data has no flag: a non-empty
    id set means that fetch succeeded. An empty set with no flag is unknown,
    so a failed TTS list is not treated as proof the model is gone.
    """
    if isinstance(data, dict) and flag in data:
        return bool(data[flag])
    return bool(model_ids(models, model_type))


@dataclass(frozen=True, slots=True)
class ResolvedSpeech:
    """Speech request that POST /audio/speech will accept."""

    text: str
    model: str
    voice: str
    response_format: str
    speed: float
    notes: tuple[str, ...]


def coerce_speech_speed(speed: object) -> float:
    """Return a speed inside the speech API range, or 1.0 when unusable."""
    try:
        value = float(speed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    if value != value:  # NaN
        return 1.0
    return min(SPEECH_SPEED_MAX, max(SPEECH_SPEED_MIN, value))


def speech_extension(response_format: str) -> str:
    """Return the file extension Home Assistant should tag the audio with."""
    token = _normalize_format(response_format)
    if token in SPEECH_FORMATS:
        return token
    return "mp3"


def resolve_tts_speech(
    *,
    text: object,
    model: object,
    voice: object = "",
    configured_voice: object = "",
    response_format: object = "",
    configured_format: object = "",
    speed: object = 1.0,
    models: object = None,
    catalog_ok: bool = False,
    fallback_model: str = "",
    fallback_voice: str = "",
) -> ResolvedSpeech:
    """Make a speech request valid for the selected TTS model.

    Assist passes the pipeline voice over the integration voice, and the
    options form has one format for every model. Venice returns HTTP 400 for
    either mismatch. A requested value is kept when the catalog says it is
    valid; otherwise the configured value, then the model default, is used.
    """
    notes: list[str] = []
    cleaned = _clean_text(text, notes)
    speed_value = coerce_speech_speed(speed)
    if _speed_was_adjusted(speed, speed_value):
        notes.append(f"speed {speed!r} adjusted to {speed_value}")

    model_id = _as_str(model)
    requested_voice = _as_str(voice)
    saved_voice = _as_str(configured_voice) or _as_str(fallback_voice)
    model_obj: dict | None = None
    chosen_model = model_id

    if catalog_ok:
        model_obj = _find_model(models, model_id) if model_id else None
        if model_id and model_obj is None:
            replacement = _first_tts_model(models, fallback_model)
            replacement_id = replacement.get("id") if replacement else None
            if isinstance(replacement_id, str) and replacement_id:
                chosen_model = replacement_id
                model_obj = replacement
                notes.append(
                    f"model {model_id!r} is not in the TTS catalog; using {chosen_model!r}"
                )
            else:
                notes.append(f"model {model_id!r} is not in the TTS catalog")

    if not chosen_model:
        chosen_model = fallback_model or model_id
        if fallback_model and not model_id:
            notes.append(f"no model selected; using {fallback_model!r}")

    if model_obj is not None:
        chosen_voice = _choose_voice(
            requested_voice, saved_voice, model_obj, notes, chosen_model
        )
        chosen_format = _choose_format(
            response_format,
            configured_format,
            model_obj,
            notes,
            chosen_model,
        )
    else:
        chosen_voice = requested_voice or saved_voice
        chosen_format = _choose_format(
            response_format or configured_format,
            "",
            None,
            notes,
            chosen_model,
        )

    return ResolvedSpeech(
        text=cleaned,
        model=chosen_model,
        voice=chosen_voice,
        response_format=chosen_format,
        speed=speed_value,
        notes=tuple(notes),
    )


def _as_str(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _speed_was_adjusted(speed: object, coerced: float) -> bool:
    if speed in (None, ""):
        return False
    try:
        value = float(speed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    return value != value or value != coerced


def _clean_text(text: object, notes: list[str]) -> str:
    raw = text if isinstance(text, str) else ""
    cleaned = raw.replace("\x00", "")
    if cleaned != raw:
        notes.append("removed null bytes from speech text")
    stripped = cleaned.strip()
    if len(stripped) > SPEECH_INPUT_LIMIT:
        notes.append(
            f"speech text truncated from {len(stripped)} to {SPEECH_INPUT_LIMIT} characters"
        )
        stripped = stripped[:SPEECH_INPUT_LIMIT]
    return stripped


def _normalize_format(value: object) -> str:
    if not isinstance(value, str):
        return ""
    token = value.strip().lower()
    if token.startswith("audio/"):
        token = token.split("/", 1)[1]
    token = token.split(";", 1)[0].strip().lstrip(".")
    return _FORMAT_ALIASES.get(token, token)




def _model_spec(model: dict) -> dict:
    spec = model.get("model_spec")
    return spec if isinstance(spec, dict) else {}


def _voice_ids(model: dict) -> list[str]:
    spec = _model_spec(model)
    raw = spec.get("voices")
    if not isinstance(raw, list) or not raw:
        raw = model.get("voice_models")
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            ids.append(item.strip())
        elif isinstance(item, dict):
            for key in ("id", "voice_id", "name"):
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    ids.append(candidate.strip())
                    break
    return ids


def _default_voice(model: dict, voices: list[str]) -> str:
    spec = _model_spec(model)
    for candidate in (model.get("default_voice"), spec.get("default_voice")):
        if isinstance(candidate, str) and candidate in voices:
            return candidate
    return voices[0] if voices else ""


def _supported_formats(model: dict) -> list[str]:
    raw = _model_spec(model).get("supported_formats")
    if not isinstance(raw, list):
        return []
    formats: list[str] = []
    for item in raw:
        token = _normalize_format(item)
        if token and token not in formats:
            formats.append(token)
    return formats


def _default_format(model: dict, formats: list[str]) -> str:
    candidate = _normalize_format(_model_spec(model).get("default_format"))
    if candidate and (not formats or candidate in formats):
        return candidate
    for preferred in _CONTAINER_PREFERENCE:
        if preferred in formats:
            return preferred
    return formats[0] if formats else "mp3"


def _find_model(models: object, model_id: str) -> dict | None:
    if not isinstance(models, list) or not model_id:
        return None
    fallback: dict | None = None
    for model in models:
        if not isinstance(model, dict) or model.get("id") != model_id:
            continue
        kind = model.get("model_type") or model.get("type")
        if kind in (None, "", "tts"):
            return model
        fallback = model
    return fallback


def _first_tts_model(models: object, preferred_id: str) -> dict | None:
    preferred = _find_model(models, preferred_id)
    if preferred is not None and _voice_ids(preferred):
        return preferred
    if not isinstance(models, list):
        return preferred
    for model in models:
        if not isinstance(model, dict):
            continue
        kind = model.get("model_type") or model.get("type")
        if kind not in (None, "", "tts"):
            continue
        if _voice_ids(model):
            return model
    return preferred


def _custom_voice_ok(model: dict) -> bool:
    spec = _model_spec(model)
    return bool(
        spec.get("supports_custom_voice_id") or model.get("supports_custom_voice_id")
    )


def _canonical_voice(value: str, voices: list[str]) -> str | None:
    if not value or not voices:
        return None
    return {voice.lower(): voice for voice in voices}.get(value.lower())


def _choose_voice(
    requested: str,
    configured: str,
    model: dict,
    notes: list[str],
    model_id: str,
) -> str:
    voices = _voice_ids(model)
    if requested.startswith("vv_"):
        return requested

    match = _canonical_voice(requested, voices)
    if match:
        if match != requested:
            notes.append(f"voice {requested!r} matched {match!r} for {model_id}")
        return match
    if requested and _custom_voice_ok(model):
        return requested

    configured_match = _canonical_voice(configured, voices)
    if requested and configured_match:
        notes.append(
            f"voice {requested!r} is not valid for {model_id}; "
            f"using configured voice {configured_match!r}"
        )
        return configured_match
    if not requested and configured_match:
        return configured_match

    if voices:
        chosen = _default_voice(model, voices)
        rejected = requested or configured
        if rejected and rejected != chosen:
            notes.append(
                f"voice {rejected!r} is not valid for {model_id}; using {chosen!r}"
            )
        elif not rejected:
            notes.append(f"no voice selected for {model_id}; using {chosen!r}")
        return chosen
    return requested or configured


def _choose_format(
    requested: object,
    configured: object,
    model: dict | None,
    notes: list[str],
    model_id: str,
) -> str:
    raw = _as_str(requested)
    token = _normalize_format(raw)
    alias_changed = bool(raw) and token != raw.lower()
    formats = _supported_formats(model) if model is not None else []
    configured_token = _normalize_format(configured)

    if not raw:
        if model is None or not formats:
            return configured_token if configured_token in SPEECH_FORMATS else "mp3"
        if configured_token in formats:
            return configured_token
        chosen = _default_format(model, formats)
        if configured_token and configured_token != chosen:
            notes.append(
                f"format {configured_token!r} is not supported by {model_id}; using {chosen!r}"
            )
        return chosen

    if model is None or not formats:
        if token in SPEECH_FORMATS:
            if alias_changed:
                notes.append(f"format {raw!r} mapped to {token!r}")
            return token
        if raw:
            notes.append(f"format {raw!r} is not a Venice speech format; using mp3")
        return "mp3"

    if token in formats:
        if alias_changed:
            notes.append(f"format {raw!r} mapped to {token!r}")
        return token

    if configured_token in formats and configured_token != token:
        notes.append(
            f"format {raw!r} is not supported by {model_id}; "
            f"using configured format {configured_token!r}"
        )
        return configured_token

    chosen = _default_format(model, formats)
    notes.append(
        f"format {raw or configured_token or 'unset'!r} is not supported by "
        f"{model_id}; using {chosen!r}"
    )
    return chosen
