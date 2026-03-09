
WHISPER_MODEL = "large-v1" # "tiny/base/small/medium/large/large-v1/large-v2/large-v3"

GEMINI_MODEL = "gemini-flash-lite-latest" 

# Đường dẫn thư mục chứa các bài hát
SONGS_DIR = r"C:\Users\Admin\Documents\demo\Nhac\song"

# Đường dẫn thư mục chứa các file lời nhạc (hiện tại hỗ trợ file txt)
LYRICS_DIR = r"C:\Users\Admin\Documents\demo\Nhac\lyrics"

# Đường dẫn thư mục chứa các file raw srt đã tạo
RAW_SRT_DIR = r"C:\Users\Admin\Documents\demo\Nhac\raw_srt"

# Đường dẫn thư mục chứa các file srt đã chỉnh sửa
SRT_DIR = r"C:\Users\Admin\Documents\demo\Nhac\srt"

GEMINI_API_KEY = "AIxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


PROMPT_SRT_MAPPING = {
        "system_prompt": """
You are a professional lyric alignment assistant.  
Align an SRT subtitle file to the reference lyrics provided by the user.

Keep timestamps and order for all block.
Check each block in SRT, if a block content in SRT has multiple lyric lines in LYRICS, simply insert line breaks for block content to match the lyric lines like LYRICS. DO NOT CHANGE start and end timestamps of block.

Skip this if it’s only ad-libs (oh, ah, etc.).

Correct minor text differences (spelling, spacing, punctuation, missing accents) using closest lyric lines.  
If not found, infer from context or leave blank.  
Keep ad-libs exactly as in the original.  

IMPORTANT:  
Delete the **first** or **last** block if its content does not appear in the lyrics and is not an ad-lib.
DO NOT CHANGE start and end timestamps of block.
Every SRT block must be separated by exactly one blank row. No exceptions.

Output only the final corrected SRT, in valid SRT format:  
(index)  
(start --> end)  
(text)  
(blank line)  
No comments or explanations.
            """,
        "user_prompt": """
--- SRT ---
{srt_text}

--- LYRICS ---
{lyrics_text}
        """
    }
