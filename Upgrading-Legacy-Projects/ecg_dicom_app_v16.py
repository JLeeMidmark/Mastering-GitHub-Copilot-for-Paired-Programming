# Version: DICOM_Gen_v16
# Added "Study & Patient" section above Channel Controls & Ordering:
# - Study ID (default: 6-char random + current time YYYYMMDDHH24MISS)
# - Study UID (numeric UID; up to 64 chars)
# - Accession Number (default: 6-char random + current time)
# - Study Description (default uses SOP, lead set, patient name & charset)
# - Patient Name (moved here)
# - Patient ID / Birthdate (moved here)
# Other functionality remains the same as v15.

import io
import os
import base64
import datetime as dt
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Optional
import random, string, uuid

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_sortables import sort_items  # Drag-and-drop list
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

def _init_log():
    if "log" not in st.session_state:
        st.session_state["log"] = []
        
def log(msg: str):
    _init_log()
    st.session_state["log"].append(str(msg))

def rand6():
    # 6 random uppercase letters/digits
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def timestamp_ymdhms():
    # YYYYMMDDHHMMSS
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d%H%M%S")

def make_uid_numeric_64():
    # Numeric UID (2.25.<int>) truncated to <= 64 chars
    n = uuid.uuid4().int
    uid = "2.25." + str(n)
    uid = uid[:64].rstrip(".")
    if len(uid) < 12:
        uid = uid + ("0" * (12 - len(uid)))
    return uid

LEAD_SETS: Dict[str, List[Tuple[str, str]]] = {
    "3":  [("I","Lead I"), ("II","Lead II"), ("III","Lead III")],
    "5":  [("I","Lead I"), ("II","Lead II"), ("III","Lead III"), ("aVR","Lead aVR"), ("V1","V1")],
    "12": [("I","Lead I"), ("II","Lead II"), ("III","Lead III"),
           ("aVR","Lead aVR"), ("aVL","Lead aVL"), ("aVF","Lead aVF"),
           ("V1","V1"), ("V2","V2"), ("V3","V3"),
           ("V4","V4"), ("V5","V5"), ("V6","V6")],
    "15R":[("I","Lead I"), ("II","Lead II"), ("III","Lead III"),
           ("aVR","Lead aVR"), ("aVL","Lead aVL"), ("aVF","Lead aVF"),
           ("V1","V1"), ("V2","V2"), ("V3","V3"),
           ("V4","V4"), ("V5","V5"), ("V6","V6"),
           ("V3R","V3R"), ("V4R","V4R"), ("V5R","V5R")],
    "15P":[("I","Lead I"), ("II","Lead II"), ("III","Lead III"),
           ("aVR","Lead aVR"), ("aVL","Lead aVL"), ("aVF","Lead aVF"),
           ("V1","V1"), ("V2","V2"), ("V3","V3"),
           ("V4","V4"), ("V5","V5"), ("V6","V6"),
           ("V7","V7"), ("V8","V8"), ("V9","V9")],
    "18": [("I","Lead I"), ("II","Lead II"), ("III","Lead III"),
           ("aVR","Lead aVR"), ("aVL","Lead aVL"), ("aVF","Lead aVF"),
           ("V1","V1"), ("V2","V2"), ("V3","V3"),
           ("V4","V4"), ("V5","V5"), ("V6","V6"),
           ("V3R","V3R"), ("V4R","V4R"), ("V5R","V5R"),
           ("V7","V7"), ("V8","V8"), ("V9","V9")],
    "15": [("I","Lead I"), ("II","Lead II"), ("III","Lead III"),
           ("aVR","Lead aVR"), ("aVL","Lead aVL"), ("aVF","Lead aVF"),
           ("V1","V1"), ("V2","V2"), ("V3","V3"),
           ("V4","V4"), ("V5","V5"), ("V6","V6"),
           ("V7","V7"), ("V8","V8"), ("V9","V9")],
}

SOP_12LEAD = "1.2.840.10008.5.1.4.1.1.9.1.1"
SOP_GENERAL = "1.2.840.10008.5.1.4.1.1.9.1.2"
SOP_AMB = "1.2.840.10008.5.1.4.1.1.9.1.3"

CHARSETS = {
    "ASCII": ("ISO_IR 6",   "ECG^Test",                "Acquisition"),
    "Latin": ("ISO_IR 100", "José^García",             "Adquisición Clínica"),
    "UTF-8": ("ISO_IR 192", "李^雷 Δ",                  "取得（試験）"),
}

MDC_CODE = {
    "I":"MDC_ECG_LEAD_I", "II":"MDC_ECG_LEAD_II", "III":"MDC_ECG_LEAD_III",
    "aVR":"MDC_ECG_LEAD_AVR", "aVL":"MDC_ECG_LEAD_AVL", "aVF":"MDC_ECG_LEAD_AVF",
    "V1":"MDC_ECG_LEAD_V1", "V2":"MDC_ECG_LEAD_V2", "V3":"MDC_ECG_LEAD_V3",
    "V4":"MDC_ECG_LEAD_V4", "V5":"MDC_ECG_LEAD_V5", "V6":"MDC_ECG_LEAD_V6",
    "V7":"MDC_ECG_LEAD_V7", "V8":"MDC_ECG_LEAD_V8", "V9":"MDC_ECG_LEAD_V9",
    "V3R":"MDC_ECG_LEAD_V3R", "V4R":"MDC_ECG_LEAD_V4R", "V5R":"MDC_ECG_LEAD_V5R",
}

def _morph_table():
    return {
        "I":   dict(P=0.12, Q=-0.06, R=0.9,  S=-0.2, T=0.35, Tinv=False),
        "II":  dict(P=0.15, Q=-0.05, R=1.0,  S=-0.2, T=0.40, Tinv=False),
        "III": dict(P=0.10, Q=-0.05, R=0.7,  S=-0.2, T=0.30, Tinv=False),
        "aVR": dict(P=-0.08, Q=0.00, R=-0.6, S=0.1,  T=-0.20, Tinv=True),
        "aVL": dict(P=0.10, Q=-0.03, R=0.6,  S=-0.1, T=0.25, Tinv=False),
        "aVF": dict(P=0.12, Q=-0.04, R=0.9,  S=-0.2, T=0.35, Tinv=False),
        "V1":  dict(P=0.05, Q=-0.02, R=0.2,  S=-1.0, T=-0.05, Tinv=True),
        "V2":  dict(P=0.07, Q=-0.03, R=0.5,  S=-0.8, T=0.05,  Tinv=False),
        "V3":  dict(P=0.08, Q=-0.03, R=0.9,  S=-0.5, T=0.20,  Tinv=False),
        "V4":  dict(P=0.08, Q=-0.03, R=1.2,  S=-0.3, T=0.35,  Tinv=False),
        "V5":  dict(P=0.08, Q=-0.03, R=1.4,  S=-0.2, T=0.45,  Tinv=False),
        "V6":  dict(P=0.08, Q=-0.03, R=1.3,  S=-0.2, T=0.40,  Tinv=False),
        "V3R": dict(P=0.05, Q=-0.02, R=0.3,  S=-0.8, T=-0.05, Tinv=True),
        "V4R": dict(P=0.05, Q=-0.02, R=0.5,  S=-0.6, T=0.00,  Tinv=False),
        "V5R": dict(P=0.05, Q=-0.02, R=0.6,  S=-0.4, T=0.05,  Tinv=False),
        "V7":  dict(P=0.08, Q=-0.02, R=0.9,  S=-0.3, T=0.30,  Tinv=False),
        "V8":  dict(P=0.08, Q=-0.02, R=0.8,  S=-0.3, T=0.25,  Tinv=False),
        "V9":  dict(P=0.08, Q=-0.02, R=0.6,  S=-0.3, T=0.20,  Tinv=False),
    }

def _skewed_gauss(x, mu, sigma_l, sigma_r, amp):
    g = np.empty_like(x)
    left = x <= mu
    g[left]  = amp * np.exp(-0.5 * ((x[left]  - mu) / sigma_l) ** 2)
    g[~left] = amp * np.exp(-0.5 * ((x[~left] - mu) / sigma_r) ** 2)
    return g

def _build_single_beat(fs: float, lead_code: str) -> np.ndarray:
    p = _morph_table().get(lead_code, _morph_table()["II"])
    PR, QRS, QT = 0.16, 0.09, 0.38
    pre_iso, post_iso = 0.08, 0.12
    dur = pre_iso + PR + QRS + QT + post_iso
    n = max(1, int(round(dur * fs)))
    x = np.linspace(0, dur, n, endpoint=False)
    t_P = pre_iso + (PR * 0.4)
    t_Q = pre_iso + PR + QRS*0.15
    t_R = pre_iso + PR + QRS*0.35
    t_S = pre_iso + PR + QRS*0.60
    t_T = pre_iso + PR + QRS + QT*0.35
    Pw = 0.045; Tw = 0.10
    Ql, Qr = 0.010, 0.014
    Rl, Rr = 0.010, 0.016
    Sl, Sr = 0.012, 0.018
    mV = lambda v: float(v) * 1000.0
    aP = mV(p["P"]); aQ = mV(p["Q"]); aR = mV(p["R"]); aS = mV(p["S"])
    aT = mV(-abs(p["T"]) if p.get("Tinv", False) else abs(p["T"]))
    y  = _skewed_gauss(x, t_P, Pw*0.8, Pw*1.2, aP)
    y += _skewed_gauss(x, t_Q, Ql, Qr, aQ)
    y += _skewed_gauss(x, t_R, Rl, Rr, aR)
    y += _skewed_gauss(x, t_S, Sl, Sr, aS)
    y += _skewed_gauss(x, t_T, Tw*0.9, Tw*1.1, aT)
    return y.astype(np.float64)

def _bandpass(signal, fs, f_lo=0.05, f_hi=150.0):
    n = len(signal)
    if n < 8: return signal
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    S = np.fft.rfft(signal)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    S *= mask
    return np.fft.irfft(S, n)

def _add_noise(signal, fs, baseline_wander_mv=0.12, powerline_hz=None, emg_std_uv=25.0):
    t = np.arange(len(signal))/fs
    y = signal.copy()
    if baseline_wander_mv and baseline_wander_mv > 0:
        y += (baseline_wander_mv * 1000.0) * np.sin(2*np.pi*0.25*t)
    if powerline_hz in (50, 60):
        y += 40.0 * np.sin(2*np.pi*powerline_hz*t)
    if emg_std_uv and emg_std_uv > 0:
        rng = np.random.default_rng(7)
        y += rng.normal(0.0, emg_std_uv, size=len(y))
    return y

def synth_ecg_uv(t: np.ndarray, hr_bpm: float, lead_code: str, add_noise=True) -> np.ndarray:
    fs = 1.0/ (t[1]-t[0]) if len(t) > 1 else 500.0
    beat = _build_single_beat(fs, lead_code)
    rr = 60.0 / max(30.0, hr_bpm)
    y = np.zeros_like(t, dtype=np.float64)
    start = 0.4; pos = start
    while pos < t[-1] + 0.3:
        i0 = int(round((pos - len(beat)/fs/2)*fs))
        i1 = i0 + len(beat)
        if i1 > 0 and i0 < len(y):
            s0 = max(0, i0); s1 = min(len(y), i1)
            b0 = s0 - i0;    b1 = b0 + (s1 - s0)
            y[s0:s1] += beat[b0:b1]
        pos += rr
    if add_noise:
        y = _add_noise(y, fs, baseline_wander_mv=0.12, powerline_hz=None, emg_std_uv=25.0)
        y = _bandpass(y, fs, 0.05, 150.0)
    return y

def container_bits_for(bits_stored: int) -> int:
    return 8 if bits_stored <= 8 else 16 if bits_stored <= 16 else 32
def sample_interpretation(container_bits: int, signed: bool) -> str:
    return {8: ("SB","UB"), 16: ("SS","US"), 32: ("SL","UL")}[container_bits][0 if signed else 1]
def dtype_for(container_bits: int, signed: bool):
    return {8:(np.int8,np.uint8),16:(np.int16,np.uint16),32:(np.int32,np.uint32)}[container_bits][0 if signed else 1]
def clip_range(bits_stored: int, signed: bool) -> Tuple[int, int]:
    return (-(1 << (bits_stored - 1)), (1 << (bits_stored - 1)) - 1) if signed else (0, (1 << bits_stored) - 1)
def unit_to_uv_per_lsb(units: str) -> float:
    u = units.lower()
    if u in ("uv","µv"): return 1.0
    if u in ("mv",):     return 1000.0
    raise ValueError("units must be 'mV' or 'uV'")
def _format_timezone_offset(iana_tz: str) -> Optional[str]:
    if not iana_tz or iana_tz == "Do not include":
        return None
    now = dt.datetime.now(ZoneInfo(iana_tz))
    offset = now.utcoffset()
    if offset is None:
        return None
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{sign}{hh:02d}{mm:02d}"

def build_ecg(
    fs_hz: float,
    bits_stored: int,
    signed: bool,
    lead_key: str,
    sop_mode: str,
    patient_id: Optional[str],
    birthdate: Optional[str],
    charset_key: str,
    omit_leads: List[str],
    partial_missing_start: Dict[str, float],
    partial_missing_end: Dict[str, float],
    per_lead_cfg: Dict[str, Tuple[str, float, float, float, float]],
    duration_s: float,
    rhythm_seconds: float,
    rhythm_lead_code: Optional[str],
    tz_offset_tag: Optional[str],
    hr_bpm: float = 60.0,
    pacer_enabled: bool = False,
    pacer_leads: Optional[List[str]] = None,
    lead_order: Optional[List[str]] = None,
    study_id: Optional[str] = None,
    accession_number: Optional[str] = None,
    study_uid: Optional[str] = None,
    study_desc_override: Optional[str] = None,
    patient_name_override: Optional[str] = None,
) -> bytes:
    if fs_hz < 100:
        raise ValueError("Sampling frequency must be >= 100 Hz.")
    if not (8 <= bits_stored <= 32):
        raise ValueError("Bits must be between 8 and 32.")
    if duration_s <= 0:
        raise ValueError("Duration must be > 0.")

    all_leads = LEAD_SETS[lead_key]
    code_to_meaning = {c:m for c,m in all_leads}
    ordered_codes = (lead_order if lead_order else [c for c,_ in all_leads])

    if sop_mode == "12-Lead":
        sop_class = SOP_12LEAD
    elif sop_mode == "General":
        sop_class = SOP_GENERAL
    elif sop_mode == "Ambulatory":
        sop_class = SOP_AMB
    else:
        raise ValueError("sop must be '12-Lead'|'General'|'Ambulatory'")

    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.FileMetaInformationVersion = b"\x00\x01"

    ds = FileDataset("mem", {}, file_meta=fm, preamble=b"\x00"*128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    char, patient_demo, study_desc_default = CHARSETS[charset_key]
    ds.SpecificCharacterSet = [char]
    if tz_offset_tag:
        ds.TimezoneOffsetFromUTC = tz_offset_tag

    now = dt.datetime.now()

    if patient_name_override and patient_name_override != "null":
        ds.PatientName = patient_name_override
    else:
        ds.PatientName = patient_demo

    if patient_id is not None and patient_id != "null":
        ds.PatientID = patient_id
    elif patient_id is None:
        ds.PatientID = "EG_DEMO"

    if birthdate is not None and birthdate != "null":
        ds.PatientBirthDate = birthdate
    elif birthdate is None:
        ds.PatientBirthDate = "19700101"

    ds.Modality = "ECG"
    if study_uid and study_uid != "null":
        ds.StudyInstanceUID = study_uid
    else:
        ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPClassUID = fm.MediaStorageSOPClassUID
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID

    if study_id and study_id != "null":
        ds.StudyID = study_id
    else:
        ds.StudyID = "ECG-STUDY"
    if accession_number and accession_number != "null":
        ds.AccessionNumber = accession_number

    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime

    lead_count = len(ordered_codes)
    sop_label = sop_mode
    pname = str(ds.PatientName)
    if study_desc_override and study_desc_override != "null":
        ds.StudyDescription = study_desc_override
    else:
        ds.StudyDescription = f"ECG {sop_label} – {lead_count} leads for {pname}"
    ds.SeriesDescription = f"ECG {lead_key}-lead"
    ds.InstitutionName = "Midmark Clinic"
    ds.ReferringPhysicianName = "John^Doe"

    n = int(round(fs_hz * duration_s))
    t = np.arange(n, dtype=np.float64) / fs_hz

    container_bits = container_bits_for(bits_stored)
    interps = sample_interpretation(container_bits, signed)
    dtype = dtype_for(container_bits, signed)
    lo, hi = clip_range(bits_stored, signed)

    omit_set = set(omit_leads)
    present_codes = [c for c in ordered_codes if c in code_to_meaning and c not in omit_set]
    log(f"present_codes (builder): {present_codes}")

    def attrs_for(code: str) -> Tuple[str,float,float,float,float]:
        return per_lead_cfg.get(code, ("mV", 0.0, 1.0, 0.0, 10.0))

    signals_by_code: Dict[str, np.ndarray] = {}
    ch_defs: List[Dataset] = []
    for idx, code in enumerate(present_codes, start=1):
        units, offset_ms, cf, baseline_counts, sens_mm_per_mV = attrs_for(code)
        sig_uv = synth_ecg_uv(t, hr_bpm=hr_bpm, lead_code=code, add_noise=True)
        sens_scale = float(sens_mm_per_mV) / 10.0
        if sens_scale != 1.0:
            sig_uv *= sens_scale

        if code in partial_missing_start and partial_missing_start[code] > 0:
            k = int(round(partial_missing_start[code] * fs_hz)); k = max(0, min(k, len(sig_uv)))
            sig_uv[:k] = 0.0
        if code in partial_missing_end and partial_missing_end[code] > 0:
            k = int(round(partial_missing_end[code] * fs_hz)); k = max(0, min(k, len(sig_uv)))
            if k > 0: sig_uv[-k:] = 0.0

        uv_per_lsb = unit_to_uv_per_lsb(units)
        counts = np.clip(np.round(sig_uv / uv_per_lsb), lo, hi).astype(dtype, copy=False)
        signals_by_code[code] = counts

        ch = Dataset()
        ch.WaveformChannelNumber = idx
        ch.ChannelLabel = code
        src = Dataset()
        src.CodeValue = MDC_CODE.get(code, code)
        src.CodingSchemeDesignator = "MDC"
        src.CodeMeaning = code
        ch.ChannelSourceSequence = Sequence([src])

        ch.ChannelSensitivity = 1.0
        unit = Dataset()
        unit.CodingSchemeDesignator = "UCUM"
        unit.CodeValue = "mV" if units.lower().startswith("m") else "uV"
        unit.CodeMeaning = "millivolt" if unit.CodeValue == "mV" else "microvolt"
        ch.ChannelSensitivityUnitsSequence = Sequence([unit])

        ch.ChannelTimeSkew = float(offset_ms) / 1000.0
        ch.ChannelSampleSkew = 0.0
        ch.ChannelBaseline = float(baseline_counts)
        ch.ChannelSensitivityCorrectionFactor = float(cf)

        ch_defs.append(ch)

    if present_codes:
        data_matrix = np.stack([signals_by_code[c] for c in present_codes], axis=0)
        interleaved = data_matrix.T.reshape(-1).astype(dtype, copy=False)
        waveform_bytes = interleaved.tobytes(order="C")
    else:
        waveform_bytes = b""

    wf_main = Dataset()
    wf_main.NumberOfWaveformChannels = len(present_codes)
    wf_main.NumberOfWaveformSamples = n
    wf_main.SamplingFrequency = float(fs_hz)
    wf_main.WaveformBitsAllocated = int(container_bits)
    wf_main.WaveformBitsStored = int(bits_stored)
    wf_main.WaveformSampleInterpretation = interps
    wf_main.ChannelDefinitionSequence = Sequence(ch_defs)
    wf_main.MultiplexGroupLabel = f"{len(present_codes)}-ch group @ {fs_hz}Hz"
    wf_main.WaveformData = waveform_bytes

    wf_items = [wf_main]

    if rhythm_seconds and rhythm_seconds > 0:
        rhythm_code = None
        if rhythm_lead_code and rhythm_lead_code in present_codes:
            rhythm_code = rhythm_lead_code
        elif "II" in present_codes:
            rhythm_code = "II"
        elif present_codes:
            rhythm_code = present_codes[0]

        if rhythm_code:
            n_r = int(round(fs_hz * rhythm_seconds))
            t_r = np.arange(n_r, dtype=np.float64) / fs_hz
            units, offset_ms, cf, baseline_counts, sens_mm_per_mV = attrs_for(rhythm_code)

            sig_uv_r = synth_ecg_uv(t_r, hr_bpm=hr_bpm, lead_code=rhythm_code, add_noise=True)
            sig_uv_r *= (float(sens_mm_per_mV) / 10.0)

            if rhythm_code in partial_missing_start and partial_missing_start[rhythm_code] > 0:
                k = int(round(partial_missing_start[rhythm_code] * fs_hz)); k = max(0, min(k, len(sig_uv_r)))
                sig_uv_r[:k] = 0.0
            if rhythm_code in partial_missing_end and partial_missing_end[rhythm_code] > 0:
                k = int(round(partial_missing_end[rhythm_code] * fs_hz)); k = max(0, min(k, len(sig_uv_r)))
                if k > 0: sig_uv_r[-k:] = 0.0

            uv_per_lsb = unit_to_uv_per_lsb(units)
            counts_r = np.clip(np.round(sig_uv_r / uv_per_lsb), lo, hi).astype(dtype, copy=False)

            ch_r = Dataset()
            ch_r.WaveformChannelNumber = 1
            ch_r.ChannelLabel = rhythm_code
            src_r = Dataset()
            src_r.CodeValue = MDC_CODE.get(rhythm_code, rhythm_code)
            src_r.CodingSchemeDesignator = "MDC"
            src_r.CodeMeaning = rhythm_code
            ch_r.ChannelSourceSequence = Sequence([src_r])
            ch_r.ChannelSensitivity = 1.0
            unit_r = Dataset()
            unit_r.CodingSchemeDesignator = "UCUM"
            unit_r.CodeValue = "mV" if units.lower().startswith("m") else "uV"
            unit_r.CodeMeaning = "millivolt" if unit_r.CodeValue == "mV" else "microvolt"
            ch_r.ChannelSensitivityUnitsSequence = Sequence([unit_r])
            ch_r.ChannelTimeSkew = float(offset_ms) / 1000.0
            ch_r.ChannelSampleSkew = 0.0
            ch_r.ChannelBaseline = float(baseline_counts)
            ch_r.ChannelSensitivityCorrectionFactor = float(cf)

            wf_r = Dataset()
            wf_r.NumberOfWaveformChannels = 1
            wf_r.NumberOfWaveformSamples = n_r
            wf_r.SamplingFrequency = float(fs_hz)
            wf_r.WaveformBitsAllocated = int(container_bits)
            wf_r.WaveformBitsStored = int(bits_stored)
            wf_r.WaveformSampleInterpretation = interps
            wf_r.ChannelDefinitionSequence = Sequence([ch_r])
            wf_r.MultiplexGroupLabel = f"Rhythm-{rhythm_code}"
            wf_r.WaveformData = counts_r.tobytes(order="C")
            wf_items.append(wf_r)

    ds.WaveformSequence = Sequence(wf_items)

    cds_items = []
    for code in present_codes:
        item = Dataset()
        item.ChannelLabel = code
        src = Dataset()
        src.CodeValue = MDC_CODE.get(code, code)
        src.CodingSchemeDesignator = "MDC"
        src.CodeMeaning = code
        item.ChannelSourceSequence = Sequence([src])
        cds_items.append(item)
    if cds_items:
        ds.ChannelDisplaySequence = Sequence(cds_items)

    try:
        cds_first = [it.ChannelLabel for it in ds.ChannelDisplaySequence][:6]
        cdef_first = [it.ChannelLabel for it in ds.WaveformSequence[0].ChannelDefinitionSequence][:6]
        log(f"CDS first labels: {cds_first}")
        log(f"ChDef first labels: {cdef_first}")
    except Exception:
        pass

    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()

st.set_page_config(page_title="Midmark ECG DICOM Generator", page_icon="🫀", layout="wide")

left, right = st.columns([0.85, 0.15])
with left:
    st.markdown("<h1 style='margin-bottom:0'>🫀 Midmark ECG DICOM Generator</h1>", unsafe_allow_html=True)
with right:
    open_help = st.button("Help", help="Open quick reference")

def show_help_modal():
    st.markdown("### How to use this app")
    st.markdown("""
- **Sampling rate (Hz)**: ≥100 (typical 500/1000).
- **BitsStored**: 8–32.
- **Lead set**: 3/5/12/15/15R/15P/18.
- **SOP class**: 12-Lead / General / Ambulatory.
- **Patient/Study fields**: You can override Study ID/UID, Accession, Description, Patient Name/ID/Birthdate.
- **Ordering**: Drag on the right; DICOM follows that order.
    """)

if open_help:
    if hasattr(st, "experimental_dialog"):
        @st.experimental_dialog("ECG DICOM Generator – Help")
        def _help():
            show_help_modal()
            st.button("Close", type="primary")
        _help()
    else:
        st.session_state["show_help_expander"] = True
if st.session_state.get("show_help_expander"):
    with st.expander("ECG DICOM Generator – Help (fallback)"):
        show_help_modal()

with st.sidebar:
    st.header("Signal & Packing")
    fs = st.number_input("Sampling rate (Hz, ≥100)", min_value=100, value=500, step=50)
    bits = st.number_input("Resolution (BitsStored, 8–32)", min_value=8, max_value=32, value=16, step=1)
    signed = st.checkbox("Signed samples", value=True)
    duration = st.number_input("Duration (s)", min_value=1.0, value=10.0, step=1.0)
    rhythm_sec = st.number_input("Rhythm strip (s, 0 to disable)", min_value=0.0, value=0.0, step=5.0)
    lead_key_preview = st.selectbox("Lead set (preview for rhythm list)", options=["3","5","12","15","15R","15P","18"], index=2)
    rhythm_options_preview = ["Auto (Lead II if present)"] + [c for c,_ in LEAD_SETS[lead_key_preview]]
    rhythm_lead_ui = st.selectbox("Rhythm lead", options=rhythm_options_preview, index=0)
    hr_bpm = st.slider("Heart rate (bpm)", min_value=40, max_value=140, value=60, step=5)

    st.header("Lead Set & SOP")
    lead_key = st.selectbox("Lead set", options=["3","5","12","15","15R","15P","18"], index=2)
    sop_mode = st.selectbox("SOP class mode", options=["12-Lead","General","Ambulatory"], index=0)

    st.header("Charset & Timezone")
    charset = st.selectbox("Character set", options=["ASCII","Latin","UTF-8"], index=0)
    tz_choice = st.selectbox(
        "Timezone (for Timezone Offset From UTC tag)",
        options=[
            "Do not include",
            "America/New_York", "America/Detroit", "America/Indiana/Indianapolis",
            "America/Chicago", "America/Winnipeg",
            "America/Denver", "America/Boise",
            "America/Phoenix",
            "America/Los_Angeles", "America/Vancouver",
            "America/Anchorage",
            "Pacific/Honolulu",
            "America/Puerto_Rico",
            "Pacific/Guam",
            "Pacific/Pago_Pago",
        ],
        index=0,
    )

# ---- New Study & Patient section ----
st.subheader("Study & Patient")

if "study_id_default" not in st.session_state:
    st.session_state.study_id_default = f"{rand6()}_{timestamp_ymdhms()}"
if "accession_default" not in st.session_state:
    st.session_state.accession_default = f"{rand6()}_{timestamp_ymdhms()}"
if "study_uid_default" not in st.session_state:
    st.session_state.study_uid_default = make_uid_numeric_64()

char_map = CHARSETS
default_name = char_map[charset][1]
if "patient_name_val" not in st.session_state:
    st.session_state.patient_name_val = default_name
elif st.session_state.get("last_charset_for_name") != charset and st.session_state.patient_name_val == st.session_state.get("last_default_name"):
    st.session_state.patient_name_val = default_name
st.session_state.last_charset_for_name = charset
st.session_state.last_default_name = default_name

c1, c2, c3 = st.columns(3)
with c1:
    study_id_ui = st.text_input("Study ID", value=st.session_state.study_id_default,
                                help="Default: 6 random chars + current time (YYYYMMDDHHMMSS)")
with c2:
    study_uid_ui = st.text_input("Study UID (numeric UID, up to 64 chars)",
                                 value=st.session_state.study_uid_default,
                                 help="Override Study Instance UID. Leave as-is to use this generated UID.")
with c3:
    accession_ui = st.text_input("Accession Number", value=st.session_state.accession_default,
                                 help="Default: 6 random chars + current time (YYYYMMDDHHMMSS)")

p1, p2, p3 = st.columns(3)
with p1:
    patient_name_ui = st.text_input("Patient Name", value=st.session_state.patient_name_val)
with p2:
    patient_id = st.text_input("Patient ID (use literal 'null' to omit)", value="EG_DEMO")
with p3:
    birthdate = st.text_input("Birthdate YYYYMMDD (use literal 'null' to omit)", value="19700101")

demo_snippet = {
    "ASCII": "ASCII",
    "Latin": "ÁÉÍÓÚ ñ",
    "UTF-8": "漢字 Δ",
}[charset]
study_desc_default = f"ECG {sop_mode} – {len(LEAD_SETS[lead_key])} leads for {patient_name_ui} [{demo_snippet}]"
study_desc_ui = st.text_input("Study Description", value=study_desc_default)

# ---------------- Channel Controls & Ordering ----------------
lead_list = [c for c,_ in LEAD_SETS[lead_key]]

def _new_channel_df():
    return pd.DataFrame({
        "Order": list(range(1, len(lead_list)+1)),
        "Lead": lead_list,
        "Omit": [False]*len(lead_list),
        "Valid seconds": [float(duration)]*len(lead_list),
        "Data missing at": ["Start"]*len(lead_list),
        "Units": ["mV"]*len(lead_list),
        "Offset ms": [0.0]*len(lead_list),
        "Correction": [1.0]*len(lead_list),
        "Baseline": [0.0]*len(lead_list),
        "Sensitivity (mm/mV)": [10.0]*len(lead_list),
        "Pacemaker": [False]*len(lead_list),
    })

if "channel_table" not in st.session_state or \
   sorted(st.session_state["channel_table"].get("Lead", [])) != sorted(lead_list) or \
   len(st.session_state["channel_table"]) != len(lead_list):
    st.session_state["channel_table"] = _new_channel_df()

st.subheader("Channel Controls & Ordering")

ROW_PX = 38
HDR_PX = 42
PAD_PX = 24
table_height = HDR_PX + len(lead_list)*ROW_PX + PAD_PX

left_col, right_col = st.columns([0.72, 0.28], gap="large")

with right_col:
    st.caption("**Drag to reorder** (top = first). Rows are aligned with the table.")
    lead_codes_current = (
        st.session_state["channel_table"]
        .sort_values("Order", kind="stable")["Lead"]
        .tolist()
    )
    new_order = sort_items(
        lead_codes_current,
        direction="vertical",
        key=f"lead_sort_{lead_key}",
    )
    if new_order and new_order != lead_codes_current:
        order_map = {code: i+1 for i, code in enumerate(new_order)}
        df = st.session_state["channel_table"].copy()
        df["Order"] = df["Lead"].map(order_map).astype(int)
        df = df.sort_values("Order", kind="stable").reset_index(drop=True)
        st.session_state["channel_table"] = df
        log(f"Drag order applied: {new_order}")

with left_col:
    st.caption("Edit **Order** (1..N) and per-lead attributes. The drag list (right) stays in sync.")
    edited = st.data_editor(
        st.session_state["channel_table"].set_index("Order"),
        key=f"editor_{lead_key}_{len(lead_list)}",
        num_rows="fixed",
        use_container_width=True,
        hide_index=False,
        height=table_height,
        column_config={
            "Lead": st.column_config.TextColumn(disabled=True, width="small"),
            "Omit": st.column_config.CheckboxColumn(width="small"),
            "Valid seconds": st.column_config.NumberColumn(min_value=0.0, max_value=float(duration), step=0.1, width="small"),
            "Data missing at": st.column_config.SelectboxColumn(options=["Start","End"], width="small"),
            "Units": st.column_config.SelectboxColumn(options=["mV","uV"], width="small"),
            "Offset ms": st.column_config.NumberColumn(step=0.5, width="small"),
            "Correction": st.column_config.NumberColumn(step=0.1, width="small"),
            "Baseline": st.column_config.NumberColumn(step=1.0, width="small"),
            "Sensitivity (mm/mV)": st.column_config.NumberColumn(step=0.5, width="small"),
            "Pacemaker": st.column_config.CheckboxColumn(width="small"),
        },
    )
    edited = edited.reset_index().rename(columns={"index":"Order"})
    edited["Order"] = pd.to_numeric(edited["Order"], errors="coerce").fillna(0).astype(int)
    edited = edited.sort_values(["Order","Lead"], kind="stable").reset_index(drop=True)
    edited["Order"] = np.arange(1, len(edited) + 1, dtype=int)
    st.session_state["channel_table"] = edited.copy()

visible_order = st.session_state["channel_table"].sort_values("Order", kind="stable")["Lead"].tolist()
log(f"Visible order: {visible_order}")

edited = st.session_state["channel_table"].copy()
omit_leads: List[str] = []
partial_missing_start: Dict[str, float] = {}
partial_missing_end: Dict[str, float] = {}
per_lead_cfg: Dict[str, Tuple[str, float, float, float, float]] = {}
pacer_leads: List[str] = []
lead_order: List[str] = []

for _, row in edited.sort_values("Order").iterrows():
    code = str(row["Lead"]).strip()
    lead_order.append(code)
    if bool(row["Omit"]):
        omit_leads.append(code)
    valid_s = max(0.0, min(float(row["Valid seconds"]), float(duration)))
    missing = float(duration) - valid_s
    if missing > 0:
        if str(row["Data missing at"]).strip().lower() == "end":
            partial_missing_end[code] = missing
        else:
            partial_missing_start[code] = missing
    units = str(row["Units"]).strip()
    off   = float(row["Offset ms"])
    corr  = float(row["Correction"])
    base  = float(row["Baseline"])
    sens  = float(row["Sensitivity (mm/mV)"])
    if (units.lower() != "mv") or off != 0.0 or corr != 1.0 or base != 0.0 or sens != 10.0:
        per_lead_cfg[code] = (units, off, corr, base, sens)
    if bool(row["Pacemaker"]):
        pacer_leads.append(code)

log(f"lead_order (from table): {lead_order}")
log(f"omit_leads: {omit_leads}")
log(f"partial_missing_start: {partial_missing_start}")
log(f"partial_missing_end: {partial_missing_end}")

st.divider()

filename = st.text_input("Suggested filename", value=f"ECG_{lead_key}_{fs}Hz_{bits}bit_lead_reordered.dcm")
save_mode = st.radio("Save mode", options=["Browser Download","Save to Server Path"], index=0, horizontal=True)
server_path = ""
if save_mode == "Save to Server Path":
    server_path = st.text_input("Server file path (absolute or relative)", value=filename)

colA, colB = st.columns([1,2])
with colA:
    generate = st.button("Generate DICOM", type="primary")
with colB:
    st.write("")

def _format_timezone_offset_ui(choice: str) -> Optional[str]:
    try:
        return _format_timezone_offset(choice)
    except Exception:
        return None

if generate:
    try:
        if fs < 100:
            st.error("Sampling frequency must be ≥ 100 Hz.")
        elif bits < 8 or bits > 32:
            st.error("BitsStored must be 8..32.")
        else:
            tz_tag = _format_timezone_offset_ui(tz_choice)
            rhythm_lead_code = None if rhythm_sec == 0 or rhythm_lead_ui.startswith("Auto") else rhythm_lead_ui

            log(f"[BUILD] Using lead_order: {lead_order}")

            blob = build_ecg(
                fs_hz=float(fs),
                bits_stored=int(bits),
                signed=bool(signed),
                lead_key=lead_key,
                sop_mode=sop_mode,
                patient_id=patient_id if patient_id != "" else None,
                birthdate=birthdate if birthdate != "" else None,
                charset_key=charset,
                omit_leads=omit_leads,
                partial_missing_start=partial_missing_start,
                partial_missing_end=partial_missing_end,
                per_lead_cfg=per_lead_cfg,
                duration_s=float(duration),
                rhythm_seconds=float(rhythm_sec),
                rhythm_lead_code=rhythm_lead_code,
                tz_offset_tag=tz_tag,
                hr_bpm=float(hr_bpm),
                pacer_enabled=(len(pacer_leads) > 0),
                pacer_leads=pacer_leads,
                lead_order=lead_order,
                study_id=study_id_ui,
                accession_number=accession_ui,
                study_uid=study_uid_ui,
                study_desc_override=study_desc_ui,
                patient_name_override=patient_name_ui,
            )

            if save_mode == "Save to Server Path":
                out_path = server_path or (filename or "ECG_Parametric.dcm")
                with open(out_path, "wb") as f:
                    f.write(blob)
                st.success(f"Saved to server path: {os.path.abspath(out_path)}")
                log(f"Saved to server path: {os.path.abspath(out_path)}")
            else:
                b64 = base64.b64encode(blob).decode()
                fname = filename or "ECG_Parametric.dcm"
                href = f'<a download="{fname}" href="data:application/dicom;base64,{b64}">Click to save “{fname}”</a>'
                st.success("DICOM generated.")
                st.markdown(href, unsafe_allow_html=True)
                log(f"Browser download prepared: {fname}")

    except Exception as e:
        st.exception(e)
        log(f"ERROR: {e!r}")

st.markdown("""---""")
_init_log()
with st.expander("🪵 Debug log (click to expand)"):
    if st.button("Clear log"):
        st.session_state["log"] = []
    for i, line in enumerate(st.session_state["log"]):
        st.text(f"{i+1:03d}: {line}")

st.markdown(
    '<div style="text-align:center;color:gray;">Developer: Rig Dubhashi but blame Dan DiLillo for issues</div>',
    unsafe_allow_html=True,
)
