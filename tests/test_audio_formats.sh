set -e
lattifai align tests/data/SA1.m4a tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.mp4 tests/data/SA1.TXT tests/data/SA1.vtt

lattifai align tests/data/SA1.ogg tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.mp3 tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.opus tests/data/SA1.TXT tests/data/SA1.vtt


lattifai align tests/data/SA1_24K.wav tests/data/SA1.TXT tests/data/SA1.vtt
lattifai align tests/data/SA1.wav tests/data/SA1.TXT tests/data/SA1.vtt

echo "All audio format tests passed."
