
# Danh sách các bài hát cần tạo srt
SRT_SONGS = [
"Bajo Tus Alas (Under Your Wings)",
"Cicatrices de Luz (Scars of Light)",
"En El Silencio Me Hablas (You Speak to Me in the Silence)",
"En El Silencio Sanas (In the Silence You Heal)",
"Tus Manos Me Sostienen (Your Hands Hold Me)",
]

# Ngôn ngữ của bài hát (theo chuẩn ISO 639-1, ví dụ: "en" cho tiếng Anh, "es" cho tiếng Tây Ban Nha, "fr" cho tiếng Pháp, "vi" cho tiếng Việt)
LANGUAGE = "es"



# Danh sách các bài hát trong playlist (tên file mp3 và srt phải trùng với tên trong danh sách này)
# Thứ tự bài hát trong capcut phải giống với thứ tự trong danh sách này để đảm bảo timestamp của srt chính xác
PLAYLIST_SONGS = [
"Bajo Tus Alas (Under Your Wings)",
"Cicatrices de Luz (Scars of Light)",
"En El Silencio Me Hablas (You Speak to Me in the Silence)",
"En El Silencio Sanas (In the Silence You Heal)",
"Tus Manos Me Sostienen (Your Hands Hold Me)",
]

PROJECT_DIR = r"C:\Users\Admin\Documents\demo\Video1"


################################################
import sys, os
from pathlib import Path
sys.path.append(Path(__file__).parent.resolve())
import config
from song_helper import SongProcessor


if __name__ == "__main__":

    mode = sys.argv[1] # "single" hoặc "playlist"

    song_processor = SongProcessor(
        list_srt_song_name=SRT_SONGS,
        playlist_song_name=PLAYLIST_SONGS,
        whisper_model=config.WHISPER_MODEL,
        language=LANGUAGE,
        correct_srt_mode=True,
        gemini_model=config.GEMINI_MODEL
    )

    if mode == "single":
        song_processor.transcribe_song()

    if mode == "playlist":
        song_processor.generate_srt_for_playlist(PROJECT_DIR)

################################################