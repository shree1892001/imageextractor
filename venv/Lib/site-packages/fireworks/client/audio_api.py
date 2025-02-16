from enum import Enum
from typing import List, Mapping, Optional, Union
from pydantic import BaseModel, Field


class Error(BaseModel, extra="forbid"):
    object: str = "error"
    type: str = "invalid_request_error"
    message: str


class ErrorResponse(BaseModel, extra="forbid"):
    error: Error = Field(default_factory=Error)


class TranscriptionRequest(BaseModel, extra="forbid"):
    model: Optional[str] = None
    vad_model: Optional[str] = None
    alignment_model: Optional[str] = None

    prompt: Optional[str] = None
    response_format: Optional[str] = None
    temperature: Optional[float] = None
    preprocessing: Optional[str] = None

    language: Optional[str] = None
    timestamp_granularities: Optional[List[str]] = None

    def to_multipart(self) -> Mapping[str, Union[str, bytes]]:
        result = {}
        for key, value in self.model_dump(exclude_none=True).items():
            if isinstance(value, (str, bytes)):
                result[key] = value
            elif isinstance(value, list):
                result[key] = ",".join(str(e) for e in value)
            else:
                result[key] = str(value)
        return result


class TranscriptionResponse(BaseModel, extra="forbid"):
    text: str


class TranscriptionWord(BaseModel, extra="forbid"):
    word: str
    start: float
    end: float
    language: str
    probability: Optional[float] = None
    hallucination_score: Optional[float] = 0.0


class TranscriptionSegment(BaseModel, extra="forbid"):
    id: int
    seek: int
    start: float
    end: float
    audio_start: float
    audio_end: float
    text: str
    language: str
    tokens: List[int]
    words: Optional[List[TranscriptionWord]] = None
    temperature: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None
    no_speech: Optional[bool] = None


class TranscriptionVerboseResponse(BaseModel, extra="forbid"):
    task: str = "transcribe"  # Not documented by returned by OAI API
    language: str
    duration: float
    text: str
    words: Optional[List[TranscriptionWord]] = None
    segments: Optional[List[TranscriptionSegment]] = None


class TranslationRequest(BaseModel, extra="forbid"):
    model: Optional[str] = None
    vad_model: Optional[str] = None
    alignment_model: Optional[str] = None

    prompt: Optional[str] = None
    response_format: Optional[str] = None
    temperature: Optional[float] = None
    preprocessing: Optional[str] = None

    def to_multipart(self) -> Mapping[str, Union[str, bytes]]:
        result = {}
        for key, value in self.model_dump(exclude_none=True).items():
            if isinstance(value, (str, bytes)):
                result[key] = value
            elif isinstance(value, list):
                result[key] = ",".join(str(e) for e in value)
            else:
                result[key] = str(value)
        return result


class AlignmentRequest(BaseModel, extra="forbid"):
    vad_model: Optional[str] = None
    alignment_model: Optional[str] = None

    text: str
    response_format: Optional[str] = None
    preprocessing: Optional[str] = None

    def to_multipart(self) -> Mapping[str, Union[str, bytes]]:
        result = {}
        for key, value in self.model_dump(exclude_none=True).items():
            if isinstance(value, (str, bytes)):
                result[key] = value
            elif isinstance(value, list):
                result[key] = ",".join(str(e) for e in value)
            else:
                result[key] = str(value)
        return result
