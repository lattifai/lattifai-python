set -e

lai alignment align -Y media.input_path=tests/data/SA1.m4a caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.TextGrid client.profile=true alignment.model_name=LattifAI/Lattice-1-Alpha
lai alignment align -Y media.input_path=tests/data/SA1.mp4 caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.vtt alignment.model_hub=modelscope caption.input.split_sentence=true -v

lai alignment align -Y media.input_path=tests/data/SA1.ogg caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.mp3 caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.opus caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.vtt

lai alignment align -Y media.input_path=tests/data/SA1.aac caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.flac caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.aiff caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.mov caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.webm caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.vtt
lai alignment align -Y media.input_path=tests/data/SA1.avi caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.vtt

lai alignment align -Y media.input_path=tests/data/SA1.mkv caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1_mkv.vtt

lai alignment align -Y media.input_path=tests/data/SA1_24K.wav caption.input.path=tests/data/SA1.TXT caption.output.path=tests/data/SA1.vtt
lai alignment align -Y tests/data/SA1.wav tests/data/SA1.TXT tests/data/SA1.vtt

echo "All audio format tests passed."
