set -e

lattifai align tests/data/SA1.m4a tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.mp4 tests/data/SA1.TXT tests/data/SA1.vtt

lattifai align tests/data/SA1.ogg tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.mp3 tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.opus tests/data/SA1.TXT tests/data/SA1.vtt

lattifai align tests/data/SA1.aac tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.flac tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.aiff tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.mov tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.webm tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.avi tests/data/SA1.TXT tests/data/SA1.vtt

lattifai align tests/data/SA1.mkv tests/data/SA1.TXT tests/data/SA1_mkv.vtt

lattifai align tests/data/SA1_24K.wav tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.wav tests/data/SA1.TXT tests/data/SA1.vtt

echo "All audio format tests passed."
