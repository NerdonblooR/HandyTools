import subprocess
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 2:
        print('Usage: python video_summary.py "<video_url>"')
        sys.exit(1)

    url = sys.argv[1]

    output_stem = Path("video")
    mp3 = output_stem.with_suffix(".mp3")
    transcript = output_stem.with_suffix(".txt")
    summary_file = Path("video_summary.md")

    print("Downloading audio...")
    run([
        "yt-dlp",
        "--cookies-from-browser", "chrome",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        url,
        "-o",
        str(output_stem),
    ])

    if not mp3.exists():
        print(f"Error: expected audio file not found: {mp3}")
        sys.exit(1)

    print("Transcribing audio with faster-whisper...")
    model = WhisperModel("base", compute_type="int8")
    segments, info = model.transcribe(str(mp3))

    with transcript.open("w", encoding="utf-8") as f:
        for segment in segments:
            text = segment.text.strip()
            if text:
                f.write(text + "\n")

    if not transcript.exists():
        print(f"Error: transcript file not found: {transcript}")
        sys.exit(1)

    print("Summarizing with LLM...")
    transcript_text = transcript.read_text(encoding="utf-8")

    prompt = (
        "Read the following video transcript and extract its argument structure.\n\n"
        "Return the result in Markdown and provide BOTH English and Chinese for every section.\n"
        "Format:\n\n"

        "## Core Thesis | 核心观点\n"
        "English:\n"
        "- ...\n"
        "中文：\n"
        "- ...\n\n"

        "## Supporting Arguments | 支持论据\n"
        "For each argument provide:\n"
        "- Argument\n"
        "- Why it supports the thesis\n"
        "- Evidence or examples used\n\n"
        "Each item must contain English and Chinese.\n\n"

        "Example format:\n"
        "### Argument 1\n"
        "English:\n"
        "- Argument: ...\n"
        "- Reasoning: ...\n"
        "- Evidence: ...\n"
        "中文：\n"
        "- 论据：...\n"
        "- 推理：...\n"
        "- 证据：...\n\n"

        "## Key Takeaway | 最重要结论\n"
        "English:\n"
        "- ...\n"
        "中文：\n"
        "- ...\n\n"

        "Rules:\n"
        "- Focus on the logical argument structure.\n"
        "- Extract the central thesis and the strongest 3–5 supporting arguments.\n"
        "- Avoid chronological summary.\n"
        "- Be concise and precise.\n\n"

        "Transcript:\n"
    )

    result = subprocess.check_output(
        ["llm", prompt + transcript_text],
        text=True,
    )

    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Video Summary\n\n")
        f.write(result)

    print(f"Transcript saved to {transcript}")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()