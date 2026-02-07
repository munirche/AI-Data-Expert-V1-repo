# Plan: AI Data Entry V1

## Context

A standalone voice-to-structured-data tool. The user speaks into a microphone, the system transcribes, extracts structured fields using an LLM, and saves the result as JSON. The tool is generic and config-driven -- the engine has no domain-specific logic; all use-case specifics (which fields to extract, prompts, vocabulary) live in a configuration file.

## Decisions Made

- **Batch transcription first** - Record, stop, then transcribe (works on desktop + iPhone)
- **Real-time transcription later** - Desktop enhancement in Phase 4
- **JSON output** - Download to user's device via browser (works local + deployed)
- **Whisper local (faster-whisper)** - HIPAA-safe, free, pre-built Windows wheels (switched from pywhispercpp which required C++ compiler)
- **Gemini for field extraction** - Free tier for development
- **Engine/use-case separation** - Engine is generic; domain defined entirely in config
- **Streamlit web app** - Browser-based UI, Python only, no HTML/CSS/JS needed
- **Deployment** - Local first, then Streamlit Community Cloud with custom domain

## Core Design Principle: Separation of Concerns

The engine and use case are kept strictly separate, just as we practiced in prior work.

| Component | Contains | Example |
|-----------|----------|---------|
| **Engine** (code) | Audio capture, transcription, LLM extraction, JSON output | `record_audio()`, `transcribe()`, `extract_fields()` |
| **Use Case** (config) | Field definitions, prompts, vocabulary, validation rules | `"patient_id"`, `"risk_assessment"`, extraction prompt |

The engine code has **no domain-specific terms**. Changing the config file switches the entire use case without touching code.

## Interface: Streamlit Web App

Streamlit turns Python scripts into web apps. No HTML, CSS, or JavaScript required. The app runs locally at `http://localhost:8501` during development.

**Why Streamlit:**
- Python only -- no web development skills needed
- Built-in audio recording widget (`st.audio_input`)
- Editable form fields for reviewing extracted data
- Works on desktop browsers and mobile (iPhone Safari)
- Free and open-source (Apache 2.0)

**App layout:**

```
+----------------------------------------------+
|  AI Data Entry                               |
|  Use case: [from config]                     |
+----------------------------------------------+
|                                              |
|  Fields to capture:        Progress: 4/7 57% |
|  [field list from config]                    |
|                                              |
|  [ Record Audio ]                            |
|                                              |
|  Transcript:                                 |
|  +--------------------------------------+    |
|  | "Patient is a 55 year old male..."   |    |
|  +--------------------------------------+    |
|                                              |
|  Extracted fields:                           |
|  +--------------------------------------+    |
|  | record_id:  [  051  ]           [x]  |    |
|  | summary:    [  55M with elev...]  [x] |   |
|  | risk:       [ moderate  v ]     [x]  |    |
|  | patterns:   [                ]  [ ]  |    |
|  | ...                                  |    |
|  +--------------------------------------+    |
|                                              |
|  [ + Add More Audio ]  [ Download JSON ]     |
|                                              |
+----------------------------------------------+
```

## Project Structure

```
AI Data Expert V1/
+-- Code/
|   +-- data_entry/                          # AI Data Entry V1 scripts
|   |   +-- ai_data_entry_v1.py             # Engine class
|   |   +-- app_data_entry_v1.py            # Streamlit web app
|   +-- expert_learning_system_v1.py         # (existing)
|   +-- expert_learning_system_v2.py         # (existing)
|   +-- cli_v2.py                            # (existing)
|   +-- export_db_v1.py                      # (existing)
|   +-- launch_lance_viewer.py               # (existing)
+-- Docs/
|   +-- Plan_for_ai_data_entry_v1.md         # Plan document
|   +-- how_it_works_ai_data_entry_v1.md     # Documentation (after implementation)
|   +-- ... (existing docs)
+-- config_data_entry.json                    # Use case configuration
+-- data_entry_output/                        # Local output (add to .gitignore)
+-- config.json                               # (existing)
+-- memory.md                                 # (existing)
+-- requirements.txt                          # (existing, add new deps)
```

## Files Created

| File | Purpose |
|------|---------|
| `Code/data_entry/ai_data_entry_v1.py` | Engine class (transcription, extraction, validation) |
| `Code/data_entry/app_data_entry_v1.py` | Streamlit web app (UI layer) |
| `config_data_entry.json` | Use case configuration (fields, prompts, Whisper settings) |
| `data_entry_output/` | Output directory for JSON files when running locally |
| `Docs/Plan_for_ai_data_entry_v1.md` | Plan document (this file) |

## Files Modified

| File | Change |
|------|--------|
| `requirements.txt` | Added `streamlit`, `faster-whisper` |
| `.gitignore` | Added `data_entry_output/`, `!config_data_entry.json` |
| `memory.md` | Added project structure entries and time log |

## Configuration Design (`config_data_entry.json`)

Defines the use case without changing engine code.

```json
{
  "use_case_name": "Display name for this use case",
  "output_directory": "./data_entry_output",

  "whisper_model": "base",
  "whisper_language": "en",

  "extraction_fields": {
    "field_name": {
      "type": "string | enum | array | object",
      "required": true/false,
      "description": "What this field represents",
      "values": ["option1", "option2"]
    }
  },

  "extraction_prompt_template": "You are a data extraction assistant...\n\nTranscript:\n{transcript}\n\nExtract into JSON:\n{schema}"
}
```

**Key points:**
- `extraction_fields` defines what the LLM should extract from the transcript
- `extraction_prompt_template` uses `{transcript}` and `{schema}` placeholders
- `whisper_model` controls accuracy vs speed (tiny/base/small/medium/large)
- Changing this file changes the entire use case

## Architecture

### Engine: `DataEntryEngine` (generic, config-driven)

```
DataEntryEngine
+-- __init__(config_path)           # Load config, init clients
+-- _load_config(path)              # Load JSON config
+-- _init_whisper()                 # Load Whisper model (lazy)
+-- transcribe(audio_bytes)         # Run Whisper on audio
+-- extract_fields(transcript)      # Send to Gemini, get structured JSON
+-- merge_fields(existing, new)     # Merge new extraction into existing fields
+-- validate_fields(data)           # Check required fields from config
+-- get_fill_progress(data)         # Return filled/total count and percentage
+-- save_json(data, filename)       # Save to local output directory
+-- to_json_string(data)            # Return JSON string (for download button)
+-- get_fields_schema()             # Return field definitions from config
```

### Streamlit App: `app_data_entry_v1.py`

The app is the UI layer. It calls engine methods and renders results.

Run with: `streamlit run Code/data_entry/app_data_entry_v1.py`

### Workflow

```
1. App loads config and displays field preview
2. User taps/clicks Record -> speaks into mic
3. User taps/clicks Stop -> audio sent to server
4. Whisper transcribes full audio -> transcript appears
5. User reviews transcript in editable text area
6. User clicks Extract -> Gemini extracts structured fields
7. Fields displayed as editable form inputs + fill progress (e.g., 4/7 fields - 57%)
8. User reviews, edits any field directly in the form
9. Option A: User clicks Download -> JSON file downloads to their device
   Option B: User clicks Add More Audio -> record again -> new fields merged -> progress updates
```

## Dependencies

| Package | Purpose | Cost |
|---------|---------|------|
| `streamlit` | Web app framework | Free |
| `faster-whisper` | Local speech-to-text (CTranslate2-based) | Free |

Already installed: `google-genai`, `numpy`, `pandas`

## Implementation Phases

### Phase 1: Working MVP (current)
- [x] Set up `Code/data_entry/` subfolder
- [x] Create `config_data_entry.json` with a sample use case
- [x] Implement engine: `transcribe`, `extract_fields`, `validate_fields`, `save_json`, `to_json_string`
- [x] Implement Streamlit app: audio recording, transcript display, extract button, editable fields
- [x] Add download button for JSON output
- [x] Batch workflow: record -> stop -> transcribe -> extract -> edit -> download
- [x] Test on desktop browser -- first trial 02/06/2026: pipeline works end to end (record -> transcribe -> extract -> fields populated)
- [ ] Test on iPhone Safari (via local network)

### Known Issues
- **Audio cut short (02/06/2026):** During first test, audio/transcription was cut short. Need to determine if it's the recording (st.audio_input) or the transcription (faster-whisper) that's truncating. Investigate next session.

### Phase 2: Polish
- [ ] Better form validation (required fields highlighted, missing fields flagged)
- [ ] Error handling (mic not found, Whisper model download, Gemini API errors)
- [ ] Save raw audio option
- [ ] Improve extraction prompt based on test results

### Phase 3: Web Deployment
- [ ] Deploy to Streamlit Community Cloud (free)
- [ ] Point custom domain via Namecheap DNS
- [ ] Test Whisper model size on cloud
- [ ] Verify full workflow on deployed version

### Phase 4: Real-Time Transcription (Desktop Enhancement)
- [ ] Stream audio in chunks, run Whisper on each chunk
- [ ] Display partial results live as user speaks
- [ ] Desktop browsers only

### Phase 5: Iterative Multi-Audio Capture
- [ ] Add "Add More Audio" button
- [ ] Implement `merge_fields` and `get_fill_progress`
- [ ] Display progress bar showing % of fields filled
- [ ] Highlight unfilled fields

### Phase 6: Future Enhancements
- [ ] Google Sheets integration
- [ ] Local LLM option (Ollama)
- [ ] Multiple output formats (CSV, etc.)
- [ ] Batch mode for multiple recordings

## HIPAA Considerations

| Pipeline Step | Where it runs | PHI exposure |
|---------------|---------------|--------------|
| Microphone capture | Local/Browser | None |
| Speech-to-text (Whisper) | Local | None |
| Field extraction (Gemini) | Cloud | Yes - development only, simulated data |
| JSON storage | Local | None |

For production with real PHI, swap Gemini for a local LLM (config change, not code change).
