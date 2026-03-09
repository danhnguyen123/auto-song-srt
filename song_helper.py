import sys
import requests
import json
from pathlib import Path
import time
from faster_whisper import WhisperModel
import os
from datetime import timedelta
from mutagen.mp3 import MP3


from pathlib import Path
sys.path.append(Path(__file__).parent.resolve())
import config as env
from whisper_helper import whisper_transcribe


class SongProcessor:

    def __init__(self, 
                 list_srt_song_name: list, 
                 playlist_song_name: list,
                 whisper_model: str, 
                 language: str = "auto", 
                 correct_srt_mode: bool = True,
                 gemini_model: str = "gemini-flash-lite-latest"
                 ):
        
        self.list_srt_song_name = list_srt_song_name
        self.playlist_song_name = playlist_song_name
        self.whisper_model = whisper_model
        self.gemini_model = gemini_model
        self.language = language
        self.correct_srt_mode = correct_srt_mode

    def transcribe_song(self): 

        asr = WhisperModel(
            self.whisper_model,
            device="auto",
            # compute_type=compute_type,
            # download_root=r"D:\hf_models"  # tuỳ chọn
        ) 

        for song_name in self.list_srt_song_name:

            song_path = Path(env.SONGS_DIR) / f"{song_name}.mp3"
            lyrics_path = Path(env.LYRICS_DIR) / f"{song_name}.txt"
            raw_srt_path = Path(env.RAW_SRT_DIR) / f"{song_name}.srt"
            edited_srt_path = Path(env.SRT_DIR) / f"{song_name}.srt"

            whisper_transcribe(
                asr=asr,
                audio=song_path,
                lang=self.language,
                out=raw_srt_path,
                )
            
            print(f"Song {song_name} transcribe successfully")  


            if self.correct_srt_mode:

                try:
                    response = self.correct_srt_with_lyrics(
                        raw_srt_path=raw_srt_path,
                        lyrics_path=lyrics_path,
                        model_id=self.gemini_model,
                    )

                    # result = response["output"][0]["content"][0]["text"].strip()
                    result = response["candidates"][0]["content"]["parts"][0]["text"]
            
                    # Lưu ra file
                    with open(edited_srt_path, "w", encoding="utf-8") as f:
                        f.write(result)

                    print(f"Song {song_name} correct transcribe successfully")  

                    # time.sleep(1)

                except Exception as e:
                    print(f"Error in correct_srt: {song_name}")
                    raise e  
                
            else:
                time.sleep(10)



    def generate_srt_for_playlist(self, output_project_path):

        current_index = 1
        time_offset = timedelta() # Bắt đầu ở mốc 00:00:00

        output_srt_path = Path(output_project_path) / "playlist.srt"

        output_timestamp_path = Path(output_project_path) / "timestamps.txt"
        
        with open(output_srt_path, 'w', encoding='utf-8') as out_srt, \
            open(output_timestamp_path, 'w', encoding='utf-8') as out_txt:
            
            for song in self.playlist_song_name:
                srt_path = Path(env.SRT_DIR) / f"{song}.srt"
                mp3_path = Path(env.SONGS_DIR) / f"{song}.mp3"
                
                if not os.path.exists(srt_path) or not os.path.exists(mp3_path):
                    print(f"Bỏ qua: Thiếu file SRT hoặc MP3 cho bài '{song}'")
                    continue

                # 1. Ghi Timestamp cho YouTube ngay lập tức trước khi cộng thêm thời gian
                yt_time = delta_to_youtube_time(time_offset)
                out_txt.write(f"{yt_time} - {song}\n")

                # 2. Lấy thời lượng MP3
                audio = MP3(mp3_path)
                audio_duration = timedelta(seconds=audio.info.length)

                # 3. Xử lý file SRT
                with open(srt_path, 'r', encoding='utf-8-sig') as in_f:
                    content = in_f.read().replace('\r\n', '\n').strip()

                blocks = content.split('\n\n')
                
                for block in blocks:
                    if not block.strip(): 
                        continue
                        
                    lines = block.split('\n')
                    if len(lines) >= 3:
                        time_line = ""
                        text_start_idx = 0
                        for i, line in enumerate(lines):
                            if ' --> ' in line:
                                time_line = line
                                text_start_idx = i + 1
                                break
                        
                        if time_line:
                            start_str, end_str = time_line.split(' --> ')
                            
                            start_td = time_to_delta(start_str) + time_offset
                            end_td = time_to_delta(end_str) + time_offset
                            
                            # Ghi sub ra file SRT tổng
                            out_srt.write(f"{current_index}\n")
                            out_srt.write(f"{delta_to_time(start_td)} --> {delta_to_time(end_td)}\n")
                            for text_line in lines[text_start_idx:]:
                                out_srt.write(f"{text_line}\n")
                            out_srt.write("\n")
                            
                            current_index += 1
                
                # Dịch chuyển mốc thời gian cho bài tiếp theo
                time_offset += audio_duration
                        

    def correct_srt_with_lyrics(self,
                                raw_srt_path, 
                                lyrics_path, 
                                model_id="gemini-flash-lite-latest", 
                                generate_content_api="generateContent",
                                retries=10,
                                backoff=0.5):
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:{generate_content_api}"

        with open(raw_srt_path, "r", encoding="utf-8") as f:
            srt_text = f.read()

        with open(lyrics_path, "r", encoding="utf-8") as f:
            lyrics_text = f.read()

        system_prompt = env.PROMPT_SRT_MAPPING["system_prompt"].strip()

        user_prompt   = env.PROMPT_SRT_MAPPING["user_prompt"].format(srt_text=srt_text, lyrics_text=lyrics_text).strip()

        payload = json.dumps({
            "system_instruction": {
            "parts": [
                {
                "text": system_prompt
                }
            ]
            },
            "contents": [
            {
                "role": "user",
                "parts": [
                {
                    "text": user_prompt
                }
                ]
            }
            ],
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingBudget": -1,
                },
                "imageConfig": {
                    "image_size": "1K"
                },
                },
            "tools": [
            {
                "googleSearch": {
                }
            },
            ],
        })
        
        headers = {
        "x-goog-api-key": f"{env.GEMINI_API_KEY}",
        "Content-Type": "application/json",
        }

        
        for attempt in range(1, retries + 1):

            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                if response.status_code >= 500:
                    raise requests.exceptions.HTTPError(f"Server error {response.status_code}: {response.text}")
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Lỗi {e} (lần {attempt}/{retries})")
                if attempt < retries:
                    time.sleep(backoff * attempt)
                else:
                    raise


def time_to_delta(t_str):
    """Chuyển chuỗi thời gian SRT thành đối tượng timedelta"""
    t_str = t_str.strip().replace('.', ',') 
    h, m, s_ms = t_str.split(':')
    s, ms = s_ms.split(',')
    return timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))

def delta_to_time(td):
    """Chuyển timedelta thành định dạng SRT chuẩn (HH:MM:SS,mmm)"""
    total_sec = int(td.total_seconds())
    ms = int(td.microseconds / 1000)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def delta_to_youtube_time(td):
    """Chuyển timedelta thành định dạng MM:SS hoặc HH:MM:SS cho YouTube"""
    total_sec = int(td.total_seconds())
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    
    # Nếu video trên 1 tiếng thì hiển thị giờ, nếu không thì chỉ hiển thị phút:giây
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"