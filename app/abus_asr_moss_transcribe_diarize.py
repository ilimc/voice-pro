"""
Transcribe-Diarize ASR implementation using MOSS API.

This module integrates the MOSS-Transcribe-Diarize model for speech-to-text
with speaker diarization capabilities.

API Documentation: https://studio.mosi.cn/docs/moss-transcribe-diarize
"""

import os
import re
import time
import base64
import requests
from typing import List, Tuple, Optional

import gradio as gr
import pysubs2
from pysubs2 import SSAFile, SSAEvent

from app.abus_path import path_change_ext
from app.abus_config import get_env

import structlog
logger = structlog.get_logger()


# API Configuration
MOSI_API_ENDPOINT = "https://studio.mosi.cn/v1/audio/transcriptions"
MOSI_MODEL_NAME = "moss-transcribe-diarize"


def get_mosi_api_key() -> str:
    """Get the API key for MOSI platform."""
    key = get_env('MOSI_API_KEY')
    if not key:
        raise ValueError(
            "MOSI_API_KEY environment variable is required. "
            "Please set it in your .env file or environment."
        )
    return key


def mosi_api_available() -> bool:
    """Check if the MOSI API is configured."""
    return get_env('MOSI_API_KEY') is not None


class DiarizeInference:
    """
    ASR inference class for MOSS-Transcribe-Diarize model.
    
    This class follows the same interface as other ASR classes in the project
    (WhisperInference, FasterWhisperInference, etc.).
    """
    
    def __init__(self):
        self.api_endpoint = get_env('MOSI_API_ENDPOINT') or MOSI_API_ENDPOINT
        self.model_name = MOSI_MODEL_NAME
        
    @staticmethod
    def available_models() -> List[str]:
        """Return available models for this ASR engine."""
        return ['moss-transcribe-diarize']
    
    @staticmethod
    def available_langs() -> List[str]:
        """Return available languages. Uses same list as other Whisper engines for UI consistency."""
        import whisper
        return sorted(list(whisper.tokenizer.LANGUAGES.values()))
    
    @staticmethod
    def available_compute_types() -> List[str]:
        """Return available compute types. Not applicable for API-based inference."""
        return ['default']

    def transcribe_file(self,
                        input_path: str,
                        params,
                        highlight_words: bool,
                        progress=None) -> List[str]:
        """
        Transcribe an audio file using the MOSS Transcribe-Diarize API.
        
        Args:
            input_path: Path to the audio file
            params: WhisperParameters object (some fields may not apply)
            highlight_words: Whether to highlight words in output
            progress: Gradio progress indicator
            
        Returns:
            List of generated subtitle file paths
        """
        try:
            if progress is not None:
                progress(0, desc="Calling Transcribe-Diarize API...")
            
            # Call the API
            result_text, elapsed_time = self._call_api(input_path, progress)
            
            if progress is not None:
                progress(0.8, desc="Generating subtitle files...")
            
            # Parse and write subtitle files
            subtitles = self._generate_subtitle_files(input_path, result_text, highlight_words)
            
            if progress is not None:
                progress(1.0, desc="Done!")
            
            logger.info(f"[abus_asr_moss_transcribe_diarize.py] transcribe_file completed in {elapsed_time:.2f}s")
            return subtitles
            
        except Exception as e:
            logger.error(f"[abus_asr_moss_transcribe_diarize.py] transcribe_file - An error occurred: {e}")
            raise
    
    def _call_api(self, audio_path: str, progress=None) -> Tuple[dict, float]:
        """
        Call the MOSS Transcribe-Diarize API with the audio file.
        
        Args:
            audio_path: Path to the audio file
            progress: Gradio progress indicator
            
        Returns:
            Tuple of (API result dict, elapsed time in seconds)
        """
        start_time = time.time()
        
        try:
            api_key = get_mosi_api_key()
        except ValueError as e:
            raise ValueError(str(e))
        
        # Read and encode the audio file as base64
        with open(audio_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine audio mime type
        ext = os.path.splitext(audio_path)[1].lower()
        mime_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.flac': 'audio/flac',
            '.m4a': 'audio/mp4',
            '.ogg': 'audio/ogg',
        }
        mime_type = mime_types.get(ext, 'audio/wav')
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "audio_data": f"data:{mime_type};base64,{audio_data}",
            "sampling_params": {
                "max_new_tokens": 16384,
                "temperature": 0.0
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        if progress is not None:
            progress(0.3, desc="Uploading audio to API...")
        
        # Make the API request
        response = requests.post(
            self.api_endpoint,
            json=payload,
            headers=headers,
            timeout=600  # 10 minutes timeout for long audio
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status {response.status_code}: {response.text}")
        
        result = response.json()
        
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    
    def _generate_subtitle_files(self, input_path: str, api_result: dict, highlight_words: bool) -> List[str]:
        """
        Parse API response and generate subtitle files.
        
        New API response format:
        {
            "asr_transcription_result": {
                "segments": [
                    {"start_s": "0.00", "end_s": "9.64", "speaker": "S01", "text": "..."}
                ],
                "full_text": "..."
            }
        }
        
        Args:
            input_path: Original audio file path (used for output naming)
            api_result: API response dict
            highlight_words: Whether to highlight words
            
        Returns:
            List of generated subtitle file paths
        """
        subtitles = []
        
        # Parse the API response into segments
        segments = self._parse_api_response(api_result)
        
        if not segments:
            logger.warning("[abus_asr_moss_transcribe_diarize.py] No segments parsed from API response")
            return subtitles
        
        # Create SSAFile for subtitle generation
        subs = SSAFile()
        
        for seg in segments:
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            speaker = seg.get('speaker', '')
            text = seg['text'].strip()
            
            # Add speaker label to text
            if speaker:
                text = f"[{speaker}] {text}"
            
            event = SSAEvent(start=start_ms, end=end_ms)
            event.text = text
            subs.append(event)
        
        # Generate output files
        folder_path = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        try:
            # SRT
            srt_path = path_change_ext(input_path, '.srt')
            subs.save(srt_path)
            subtitles.append(srt_path)
            
            # VTT
            vtt_path = path_change_ext(input_path, '.vtt')
            subs.save(vtt_path)
            subtitles.append(vtt_path)
            
            # TXT (plain text without timestamps)
            txt_path = path_change_ext(input_path, '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                for seg in segments:
                    speaker = seg.get('speaker', '')
                    text = seg['text'].strip()
                    if speaker:
                        f.write(f"[{speaker}] {text}\n")
                    else:
                        f.write(f"{text}\n")
            subtitles.append(txt_path)
            
            # TSV (tab-separated: start, end, speaker, text)
            tsv_path = path_change_ext(input_path, '.tsv')
            with open(tsv_path, 'w', encoding='utf-8') as f:
                f.write("start\tend\tspeaker\ttext\n")
                for seg in segments:
                    f.write(f"{seg['start']:.2f}\t{seg['end']:.2f}\t{seg.get('speaker', '')}\t{seg['text'].strip()}\n")
            subtitles.append(tsv_path)
            
            # JSON
            import json
            json_path = path_change_ext(input_path, '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({'segments': segments}, f, ensure_ascii=False, indent=2)
            subtitles.append(json_path)
            
        except Exception as e:
            logger.error(f"[abus_asr_moss_transcribe_diarize.py] Error writing subtitle files: {e}")
        
        return subtitles
    
    def _parse_api_response(self, api_result: dict) -> List[dict]:
        """
        Parse the API response dict into structured segments.
        
        New API format:
        {
            "asr_transcription_result": {
                "segments": [
                    {"start_s": "0.00", "end_s": "9.64", "speaker": "S01", "text": "..."}
                ]
            }
        }
        
        Args:
            api_result: API response dict
            
        Returns:
            List of segment dictionaries with keys: start, end, speaker, text
        """
        segments = []
        
        try:
            # Get segments from new JSON format
            asr_result = api_result.get('asr_transcription_result', {})
            raw_segments = asr_result.get('segments', [])
            
            for seg in raw_segments:
                start_time = self._parse_timestamp(seg.get('start_s', '0'))
                end_time = self._parse_timestamp(seg.get('end_s', '0'))
                speaker = seg.get('speaker', '')
                text = seg.get('text', '')
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'speaker': speaker,
                    'text': text
                })
        except Exception as e:
            logger.error(f"[abus_asr_moss_transcribe_diarize.py] Error parsing API response: {e}")
        
        return segments
    
    def _parse_timestamp(self, ts: str) -> float:
        """
        Parse a timestamp string to seconds.
        
        Supports formats:
        - MM:SS.ss (e.g., 00:01.08)
        - SS.ss (e.g., 5.54)
        """
        if ':' in ts:
            parts = ts.split(':')
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(ts)
