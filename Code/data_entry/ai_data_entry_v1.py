"""
AI Data Entry V1 - Generic Data Entry Engine

Config-driven engine: transcribes audio, extracts structured fields via LLM,
validates, and exports JSON. No domain-specific logic -- all use-case specifics
live in config_data_entry.json.
"""

import io
import json
import os
from datetime import datetime

from google import genai


class DataEntryEngine:
    """Config-driven engine for voice-to-structured-data conversion."""

    def __init__(self, config_path="./config_data_entry.json"):
        """Load config and initialize clients."""
        self.config = self._load_config(config_path)
        self.whisper_model = None

        api_key = os.environ.get("GEMINI_API_KEY")
        self.llm_client = genai.Client(api_key=api_key)

    def _load_config(self, path):
        """Load JSON configuration file."""
        with open(path, "r") as f:
            return json.load(f)

    def _init_whisper(self):
        """Load Whisper model on first use (lazy initialization)."""
        if self.whisper_model is not None:
            return

        from faster_whisper import WhisperModel

        model_name = self.config.get("whisper_model", "base")
        language = self.config.get("whisper_language", "en")
        self.whisper_model = WhisperModel(model_name, device="cpu")
        self._whisper_language = language

    def transcribe(self, audio_bytes):
        """Transcribe audio bytes to text using Whisper.

        Args:
            audio_bytes: Raw audio data (WAV format from st.audio_input).

        Returns:
            Transcribed text string.
        """
        self._init_whisper()

        # Write audio bytes to a temp file (faster-whisper reads files)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()

        try:
            segments, _ = self.whisper_model.transcribe(
                tmp.name, language=self._whisper_language
            )
            transcript = " ".join(segment.text.strip() for segment in segments)
        finally:
            os.unlink(tmp.name)

        return transcript

    def extract_fields(self, transcript):
        """Send transcript to LLM and extract structured fields.

        Args:
            transcript: Text transcript to extract fields from.

        Returns:
            Dictionary of extracted field values.
        """
        fields = self.config["extraction_fields"]

        # Build schema description for the LLM
        schema_lines = []
        for name, definition in fields.items():
            line = f'- "{name}" ({definition["type"]}): {definition["description"]}'
            if definition["type"] == "enum" and "values" in definition:
                line += f' Allowed values: {definition["values"]}'
            if "format" in definition:
                line += f' Format: {definition["format"]}'
            schema_lines.append(line)
        schema_text = "\n".join(schema_lines)

        prompt = self.config["extraction_prompt_template"].format(
            transcript=transcript,
            schema=schema_text,
        )

        response = self.llm_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        # Parse JSON from response
        raw_text = response.text.strip()

        # Remove markdown code blocks if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            raw_text = "\n".join(lines[1:-1])

        try:
            extracted = json.loads(raw_text)
        except json.JSONDecodeError:
            extracted = {"_raw_response": raw_text, "_parse_error": True}

        return extracted

    def merge_fields(self, existing, new):
        """Merge newly extracted fields into existing data.

        New values fill empty/null fields. Non-empty fields keep their value.

        Args:
            existing: Current field values dictionary.
            new: Newly extracted field values dictionary.

        Returns:
            Merged dictionary.
        """
        merged = dict(existing)

        for key, value in new.items():
            if value is None:
                continue

            current = merged.get(key)

            # Fill if current is empty/null/empty-list
            if current is None or current == "" or current == []:
                merged[key] = value

        return merged

    def validate_fields(self, data):
        """Check which required fields are filled.

        Args:
            data: Dictionary of field values.

        Returns:
            Dictionary with 'valid' (bool), 'missing' (list of missing required
            field names), and 'filled' (list of filled field names).
        """
        fields = self.config["extraction_fields"]
        missing = []
        filled = []

        for name, definition in fields.items():
            value = data.get(name)
            is_filled = value is not None and value != "" and value != []

            if is_filled:
                filled.append(name)
            elif definition.get("required", False):
                missing.append(name)

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "filled": filled,
        }

    def get_fill_progress(self, data):
        """Return count and percentage of filled fields.

        Args:
            data: Dictionary of field values.

        Returns:
            Dictionary with 'filled', 'total', and 'percentage'.
        """
        fields = self.config["extraction_fields"]
        total = len(fields)
        filled = 0

        for name in fields:
            value = data.get(name)
            if value is not None and value != "" and value != []:
                filled += 1

        percentage = round((filled / total) * 100) if total > 0 else 0

        return {"filled": filled, "total": total, "percentage": percentage}

    def save_json(self, data, filename=None):
        """Save data as JSON to the output directory.

        Args:
            data: Dictionary to save.
            filename: Optional filename. Auto-generated if not provided.

        Returns:
            Path to the saved file.
        """
        output_dir = self.config.get("output_directory", "./data_entry_output")
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"entry_{timestamp}.json"

        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def to_json_string(self, data):
        """Convert data dictionary to formatted JSON string.

        Args:
            data: Dictionary to convert.

        Returns:
            Formatted JSON string.
        """
        return json.dumps(data, indent=2)

    def get_fields_schema(self):
        """Return field definitions from config.

        Returns:
            Dictionary of field name -> field definition.
        """
        return self.config["extraction_fields"]
