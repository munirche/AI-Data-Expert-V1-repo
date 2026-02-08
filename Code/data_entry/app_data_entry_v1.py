"""
AI Data Entry V1 - Streamlit Web App

UI layer for voice-to-structured-data workflow.
Run with: streamlit run Code/data_entry/app_data_entry_v1.py
"""

import json
import os
import sys

import streamlit as st

# Add project root to path so engine can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(project_root, "Code", "data_entry"))

from ai_data_entry_v1 import DataEntryEngine


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(page_title="Virtual Medical Assistant", layout="centered")

# Hide Streamlit header/footer and reduce top padding
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none; }
    footer { display: none; }
    #MainMenu { display: none; }
    .block-container { padding-top: 0.5rem; }
    hr { margin-top: 0.5rem; margin-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# INITIALIZATION
# =============================================================================

@st.cache_resource
def load_engine():
    """Load the engine once and cache it across reruns."""
    config_path = os.path.join(project_root, "config_data_entry.json")
    return DataEntryEngine(config_path)


engine = load_engine()

# Session state defaults
if "fields_data" not in st.session_state:
    st.session_state.fields_data = {}
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "full_transcript" not in st.session_state:
    st.session_state.full_transcript = ""
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0


# =============================================================================
# HEADER
# =============================================================================

st.title("Virtual Medical Assistant")
st.markdown(
    f"<p style='margin-top: -1rem; margin-bottom: -0.5rem;'>{engine.config['use_case_name']}</p>",
    unsafe_allow_html=True,
)

st.divider()


# =============================================================================
# FIELD PREVIEW AND PROGRESS
# =============================================================================

schema = engine.get_fields_schema()
progress = engine.get_fill_progress(st.session_state.fields_data)

col_fields, col_progress = st.columns([2, 1])

with col_fields:
    has_data = len(st.session_state.fields_data) > 0
    chips = ""
    for name, d in schema.items():
        if not has_data:
            # No extraction yet: neutral style
            bg = "transparent"
            border = "#ccc"
            color = "inherit"
        else:
            value = st.session_state.fields_data.get(name)
            is_filled = value is not None and value != "" and value != []
            if is_filled:
                bg = "#d4edda"
                border = "#28a745"
                color = "#155724"
            else:
                bg = "#f8d7da"
                border = "#dc3545"
                color = "#721c24"
        chips += (
            f"<span style='display:inline-block; padding:0.05rem 0.5rem; margin:0.1rem; "
            f"background:{bg}; border:1px solid {border}; border-radius:1rem; "
            f"font-size:0.85rem; color:{color};'>"
            f"{d.get('short_description', d['description'])}</span>"
        )
    st.markdown(f"**Fields to capture:**<br>{chips}", unsafe_allow_html=True)

with col_progress:
    if progress["total"] > 0:
        st.metric(
            "Progress",
            f"{progress['filled']}/{progress['total']}",
            f"{progress['percentage']}%",
        )


# =============================================================================
# AUDIO RECORDING
# =============================================================================

# Set to True to show transcript for review before extracting
SHOW_TRANSCRIPT = False

st.subheader("Record Audio")

audio_data = st.audio_input("Tap the microphone to record", key=f"audio_{st.session_state.audio_key}")

if audio_data is not None:
    audio_bytes = audio_data.getvalue()

    if SHOW_TRANSCRIPT:
        # Two-step mode: transcribe first, then extract separately
        if st.button("Transcribe"):
            with st.spinner("Transcribing audio..."):
                try:
                    transcript = engine.transcribe(audio_bytes)
                    st.session_state.transcript = transcript

                    if st.session_state.full_transcript:
                        st.session_state.full_transcript += "\n\n" + transcript
                    else:
                        st.session_state.full_transcript = transcript

                    st.rerun()
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
    else:
        # Auto mode: transcribe and extract as soon as audio is recorded
        with st.spinner("Transcribing and extracting fields..."):
            try:
                transcript = engine.transcribe(audio_bytes)
                st.session_state.transcript = transcript

                if st.session_state.full_transcript:
                    st.session_state.full_transcript += "\n\n" + transcript
                else:
                    st.session_state.full_transcript = transcript

                extracted = engine.extract_fields(st.session_state.full_transcript)

                if "_parse_error" in extracted:
                    st.error("LLM returned invalid JSON. Raw response shown below.")
                    st.code(extracted.get("_raw_response", ""))
                else:
                    st.session_state.fields_data = engine.merge_fields(
                        st.session_state.fields_data, extracted
                    )
                    st.session_state.audio_key += 1
                    st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")


# =============================================================================
# TRANSCRIPT DISPLAY (only shown when SHOW_TRANSCRIPT is enabled)
# =============================================================================

if SHOW_TRANSCRIPT and st.session_state.full_transcript:
    st.subheader("Transcript")
    edited_transcript = st.text_area(
        "Review and edit transcript before extracting",
        value=st.session_state.full_transcript,
        height=150,
    )
    st.session_state.full_transcript = edited_transcript

    if st.button("Extract Fields", type="primary"):
        with st.spinner("Extracting fields..."):
            try:
                extracted = engine.extract_fields(st.session_state.full_transcript)

                if "_parse_error" in extracted:
                    st.error("LLM returned invalid JSON. Raw response shown below.")
                    st.code(extracted.get("_raw_response", ""))
                else:
                    st.session_state.fields_data = engine.merge_fields(
                        st.session_state.fields_data, extracted
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Extraction failed: {e}")


# =============================================================================
# EDITABLE FIELDS
# =============================================================================

if st.session_state.fields_data:
    st.subheader("Extracted Fields")

    updated_data = {}

    for field_name, definition in schema.items():
        value = st.session_state.fields_data.get(field_name)
        field_type = definition.get("type", "string")
        required = definition.get("required", False)
        desc = definition.get("description", field_name)
        label = f"{desc} *" if required else desc

        if field_type == "enum":
            options = definition.get("values", [])
            # Add empty option at start so user can clear selection
            all_options = [""] + options
            current_index = 0
            if value in options:
                current_index = all_options.index(value)
            selected = st.selectbox(label, all_options, index=current_index)
            updated_data[field_name] = selected if selected else None

        elif field_type == "array":
            # Display array as comma-separated editable text
            if isinstance(value, list):
                display_value = ", ".join(str(v) for v in value)
            else:
                display_value = str(value) if value else ""
            edited = st.text_area(label, value=display_value, height=68)
            if edited.strip():
                updated_data[field_name] = [
                    item.strip() for item in edited.split(",") if item.strip()
                ]
            else:
                updated_data[field_name] = []

        elif field_type == "string" and field_name in ("analysis", "summary"):
            # Longer text fields get a text_area
            display_value = str(value) if value else ""
            edited = st.text_area(label, value=display_value, height=100)
            updated_data[field_name] = edited if edited else None

        else:
            # Default: single-line text input
            display_value = str(value) if value else ""
            edited = st.text_input(label, value=display_value)
            updated_data[field_name] = edited if edited else None

    # Update session state with edits
    st.session_state.fields_data = updated_data

    # Refresh progress after edits
    progress = engine.get_fill_progress(st.session_state.fields_data)

    st.divider()

    # Validation status
    validation = engine.validate_fields(st.session_state.fields_data)
    if not validation["valid"]:
        st.warning(f"Missing required fields: {', '.join(validation['missing'])}")

    # ==========================================================================
    # ACTION BUTTONS
    # ==========================================================================

    col_download, col_clear = st.columns(2)

    with col_download:
        json_string = engine.to_json_string(st.session_state.fields_data)
        st.download_button(
            label="Download",
            data=json_string,
            file_name="data_entry.txt",
            mime="text/plain",
        )

    with col_clear:
        if st.button("Clear All"):
            st.session_state.fields_data = {}
            st.session_state.transcript = ""
            st.session_state.full_transcript = ""
            st.rerun()
