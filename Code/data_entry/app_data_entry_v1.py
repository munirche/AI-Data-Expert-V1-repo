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

st.set_page_config(page_title="AI Data Entry", layout="centered")


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


# =============================================================================
# HEADER
# =============================================================================

st.title("AI Data Entry")
st.caption(f"Use case: {engine.config['use_case_name']}")

st.divider()


# =============================================================================
# FIELD PREVIEW AND PROGRESS
# =============================================================================

schema = engine.get_fields_schema()
progress = engine.get_fill_progress(st.session_state.fields_data)

col_fields, col_progress = st.columns([2, 1])

with col_fields:
    lines = ["**Fields to capture:**"]
    for name, definition in schema.items():
        req = " *" if definition.get("required") else ""
        lines.append(f"- {definition['description']}{req}")
    st.markdown("\n".join(lines))

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

st.subheader("Record Audio")

audio_data = st.audio_input("Tap the microphone to record")

if audio_data is not None:
    audio_bytes = audio_data.getvalue()

    if st.button("Transcribe"):
        with st.spinner("Transcribing audio..."):
            try:
                transcript = engine.transcribe(audio_bytes)
                st.session_state.transcript = transcript

                # Accumulate transcript across recordings
                if st.session_state.full_transcript:
                    st.session_state.full_transcript += "\n\n" + transcript
                else:
                    st.session_state.full_transcript = transcript

                st.rerun()
            except Exception as e:
                st.error(f"Transcription failed: {e}")


# =============================================================================
# TRANSCRIPT DISPLAY
# =============================================================================

if st.session_state.full_transcript:
    st.subheader("Transcript")
    edited_transcript = st.text_area(
        "Review and edit transcript before extracting",
        value=st.session_state.full_transcript,
        height=150,
    )
    st.session_state.full_transcript = edited_transcript

    # Extract button
    if st.button("Extract Fields", type="primary"):
        with st.spinner("Extracting fields..."):
            try:
                extracted = engine.extract_fields(st.session_state.full_transcript)

                if "_parse_error" in extracted:
                    st.error("LLM returned invalid JSON. Raw response shown below.")
                    st.code(extracted.get("_raw_response", ""))
                else:
                    # Merge with existing fields (new fills empty slots)
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
        label = f"{field_name} *" if required else field_name

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

    # Show Save Locally only when running on local machine
    is_local = not os.environ.get("STREAMLIT_SHARING_MODE")

    if is_local:
        col_download, col_save, col_clear = st.columns(3)
    else:
        col_download, col_clear = st.columns(2)

    with col_download:
        json_string = engine.to_json_string(st.session_state.fields_data)
        st.download_button(
            label="Download JSON",
            data=json_string,
            file_name="data_entry.json",
            mime="application/json",
        )

    if is_local:
        with col_save:
            if st.button("Save Locally"):
                try:
                    filepath = engine.save_json(st.session_state.fields_data)
                    st.success(f"Saved to {filepath}")
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with col_clear:
        if st.button("Clear All"):
            st.session_state.fields_data = {}
            st.session_state.transcript = ""
            st.session_state.full_transcript = ""
            st.rerun()
