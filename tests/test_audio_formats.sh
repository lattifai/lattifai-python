set -e

lai alignment align -Y media.input_path=tests/data/SA1.m4a caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.TextGrid
lai alignment align -Y media.input_path=tests/data/SA1.mp4 caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.vtt alignment.model_hub=modelscope caption.split_sentence=true

lai alignment align -Y media.input_path=tests/data/SA1.ogg caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.mp3 caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.opus caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.vtt

lai alignment align -Y media.input_path=tests/data/SA1.aac caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.flac caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.aiff caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.mov caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.webm caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.avi caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.vtt

lai alignment align -Y media.input_path=tests/data/SA1.mkv caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1_mkv.vtt

lai alignment align -Y media.input_path=tests/data/SA1_24K.wav caption.input_path=tests/data/SA1.TXT caption.output_path=tests/data/SA1.vtt
lai alignment align -Y tests/data/SA1.wav tests/data/SA1.TXT tests/data/SA1.vtt

echo "All audio format tests passed."
