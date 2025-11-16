set -e

lai alignment align media.input_path=tests/data/SA1.m4a subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.TextGrid -W
lai alignment align media.input_path=tests/data/SA1.mp4 subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt

lai alignment align media.input_path=tests/data/SA1.ogg subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt
lai alignment align media.input_path=tests/data/SA1.mp3 subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt
lai alignment align media.input_path=tests/data/SA1.opus subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt

lai alignment align media.input_path=tests/data/SA1.aac subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt
lai alignment align media.input_path=tests/data/SA1.flac subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt
lai alignment align media.input_path=tests/data/SA1.aiff subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt
lai alignment align media.input_path=tests/data/SA1.mov subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt
lai alignment align media.input_path=tests/data/SA1.webm subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt
lai alignment align media.input_path=tests/data/SA1.avi subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt

lai alignment align media.input_path=tests/data/SA1.mkv subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1_mkv.vtt

lai alignment align media.input_path=tests/data/SA1_24K.wav subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt
lai alignment align media.input_path=tests/data/SA1.wav subtitle.input_path=tests/data/SA1.TXT subtitle.output_path=tests/data/SA1.vtt

echo "All audio format tests passed."
