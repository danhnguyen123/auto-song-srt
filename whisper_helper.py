import os
import re
from pathlib import Path
from faster_whisper import WhisperModel
import time

# ======= Khuyến nghị cho Windows: tránh lỗi symlink (WinError 1314) =======
# os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
# Tuỳ chọn: đổi nơi cache để quản lý dung lượng/tốc độ
# os.environ["HF_HOME"] = r"D:\hf_cache"

# --------- util for time formatting ----------
def fmt(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = int(t) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# --------- write SRT ----------
def write_srt(segments, out_path):
    # SỬA: đảm bảo out_path là Path, tránh f/raw string
    out_path = Path(out_path)
    lines = []
    for i, seg in enumerate(segments, 1):
        # seg ở đây là dict {"start": float, "end": float, "text": str}
        start = max(seg.get("start", 0.0), 0.0)
        end   = max(seg.get("end", start + 0.5), 0.0)
        text  = (seg.get("text") or "").strip()
        lines += [str(i), f"{fmt(start)} --> {fmt(end)}", text, ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")

# --------- text parsing helpers ----------
_word_re = re.compile(r"\w+", flags=re.UNICODE)

def _first_word_after_first_bracket(txt: str) -> str | None:
    idx = txt.find("]")
    if idx == -1:
        m = _word_re.search(txt)
        return m.group(0).casefold() if m else None
    m = _word_re.search(txt, idx + 1)
    return m.group(0).casefold() if m else None

def _last_word(txt: str) -> str | None:
    ms = list(_word_re.finditer(txt))
    if not ms:
        return None
    return ms[-1].group(0).casefold()

def _first_word_of_segment(seg_text: str) -> str | None:
    m = _word_re.search(seg_text or "")
    return m.group(0).casefold() if m else None

def _last_word_of_segment(seg_text: str) -> str | None:
    ms = list(_word_re.finditer(seg_text or ""))
    if not ms:
        return None
    return ms[-1].group(0).casefold()

def _adjust_segments_with_lyrics(segments: list[dict], lyrics_txt_path: str) -> list[dict]:
    """
    Áp dụng hai quy tắc:
    1) So sánh từ đầu (txt sau ']' đầu tiên) với từ đầu của segment[0]. Nếu khác -> xóa segment[0].
    2) So sánh từ cuối (txt) với từ cuối của segment[-1]. Nếu khác -> xóa segment[-1].
    """
    try:
        # SỬA: bỏ fr-string; dùng Path nhất quán
        lyrics_txt_path = Path(lyrics_txt_path)
        txt = lyrics_txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Cannot read lyrics_txt '{lyrics_txt_path}': {e}. Skip alignment.")
        return segments

    segs = list(segments)  # copy nông

    # --- Rule 1: đầu ---
    if segs:
        txt_first = _first_word_after_first_bracket(txt)
        srt_first = _first_word_of_segment(segs[0].get("text", ""))
        print(f"[DEBUG] txt_first={txt_first} | srt_first={srt_first}")
        if txt_first and srt_first and (txt_first != srt_first):
            print("[INFO] First words mismatch -> drop first SRT segment")
            segs.pop(0)

    # --- Rule 2: cuối ---
    if segs:
        txt_last = _last_word(txt)
        srt_last = _last_word_of_segment(segs[-1].get("text", ""))
        print(f"[DEBUG] txt_last={txt_last} | srt_last={srt_last}")
        if txt_last and srt_last and (txt_last != srt_last):
            print("[INFO] Last words mismatch -> drop last SRT segment")
            segs.pop()

    return segs

# --------- main transcribe (phiên bản faster-whisper) ----------
def whisper_transcribe(
    asr: WhisperModel = None,
    audio: str = "",
    lang: str = "auto",
    out: str | None = None,
    # temp: float = 0.0,
    lyrics_txt: str | None = None,
    # compute_type: str | None = None,
    beam_size: int = 5,
    # vad_filter: bool = True,
    
):
    """
    Drop-in thay thế cho hàm cũ dùng openai-whisper, nhưng backend là faster-whisper.
    - Trả SRT y như trước
    - Giữ tham số tương tự, bổ sung compute_type/beam_size/vad_filter.
    """

    # SỬA: chuẩn hoá path đầu vào/đầu ra bằng Path, không dùng f/raw string
    audio_path = Path(audio)

    # Device & compute_type
    # if device == "auto":
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    # if compute_type is None:
    #     compute_type = "float16" if device == "cuda" else "int8"

    # print(f"[INFO] device={device}, model={model}, lang={lang}, compute_type=None")

    # Khởi tạo model
    # asr = WhisperModel(
    #     model,
    #     device=device,
    #     # compute_type=compute_type,
    #     # download_root=r"D:\hf_models"  # tuỳ chọn
    # )

    # Map ngôn ngữ
    language = None if (not lang or lang.lower() == "auto") else lang

    # Gọi transcribe (chuyển Path -> str)
    segments_gen, info = asr.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        # vad_filter=False,
        # temperature=temp,
        # condition_on_previous_text=True,
        # word_timestamps=False,
    )

    # faster-whisper trả generator Segment; chuyển sang list[dict] tương thích
    segments_list: list[dict] = []
    for seg in segments_gen:
        segments_list.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": (seg.text or "").strip(),
        })

    print(f"[INFO] detected_language={info.language}")

    # for seg in segments_list:
    #     print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")

    # print("\n===== TRANSCRIPT =====\n")
    # full_text = " ".join(s["text"] for s in segments_list).strip()
    # print(full_text)
    # print("\n======================\n")

    # --- Optional alignment with lyrics .txt ---
    if lyrics_txt:
        print(f"[INFO] Aligning SRT with lyrics_txt = {lyrics_txt}")
        segments_list = _adjust_segments_with_lyrics(segments_list, lyrics_txt)

    # SỬA: tạo out_path an toàn bằng Path
    out_path = Path(out) if out else audio_path.with_suffix(".srt")
    write_srt(segments_list, out_path)
    print(f"[OK] Wrote SRT: {out_path}")

    # time.sleep(10)