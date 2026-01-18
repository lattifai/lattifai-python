"""Test Caption I/O with in-memory data support."""

import io

from lattifai.caption import Caption


def test_read_from_stringio():
    """Test reading caption from StringIO."""
    srt_content = """1
00:00:00,000 --> 00:00:02,000
Hello world

2
00:00:02,500 --> 00:00:05,000
This is a test
"""
    buffer = io.StringIO(srt_content)
    caption = Caption.read(buffer, format="srt")

    assert len(caption) == 2
    assert caption.supervisions[0].text == "Hello world"
    assert caption.supervisions[1].text == "This is a test"
    assert caption.source_format == "srt"
    print("✅ test_read_from_stringio passed")


def test_read_from_bytesio():
    """Test reading caption from BytesIO."""
    srt_content = b"""1
00:00:00,000 --> 00:00:02,000
Hello world

2
00:00:02,500 --> 00:00:05,000
This is a test
"""
    buffer = io.BytesIO(srt_content)
    caption = Caption.read(buffer, format="srt")

    assert len(caption) == 2
    assert caption.supervisions[0].text == "Hello world"
    assert caption.supervisions[1].text == "This is a test"
    print("✅ test_read_from_bytesio passed")


def test_from_string():
    """Test creating caption from string."""
    vtt_content = """WEBVTT

00:00:00.000 --> 00:00:02.000
Hello world

00:00:02.500 --> 00:00:05.000
This is a test
"""
    caption = Caption.from_string(vtt_content, format="vtt")

    assert len(caption) == 2
    assert caption.supervisions[0].text == "Hello world"
    assert caption.source_format == "vtt"
    print("✅ test_from_string passed")


def test_write_to_bytesio():
    """Test writing caption to BytesIO."""
    srt_content = """1
00:00:00,000 --> 00:00:02,000
Hello world
"""
    caption = Caption.from_string(srt_content, format="srt")

    buffer = io.BytesIO()
    result = caption.write(buffer)

    assert isinstance(result, bytes)
    assert b"Hello world" in result

    # Verify buffer can be read
    buffer.seek(0)
    content = buffer.read()
    assert b"Hello world" in content
    print("✅ test_write_to_bytesio passed")


def test_to_bytes():
    """Test converting caption to bytes."""
    srt_content = """1
00:00:00,000 --> 00:00:02,000
Hello world

2
00:00:02,500 --> 00:00:05,000
This is a test
"""
    caption = Caption.from_string(srt_content, format="srt")

    data = caption.to_bytes()

    assert isinstance(data, bytes)
    assert b"Hello world" in data
    assert b"This is a test" in data
    print("✅ test_to_bytes passed")


def test_round_trip_stringio():
    """Test round-trip conversion through StringIO."""
    original_content = """1
00:00:00,000 --> 00:00:02,000
Hello world

2
00:00:02,500 --> 00:00:05,000
This is a test
"""
    # Read from string
    caption = Caption.from_string(original_content, format="srt")

    # Write to bytes
    data = caption.to_bytes()

    # Read back from bytes
    buffer = io.BytesIO(data)
    caption2 = Caption.read(buffer, format="srt")

    assert len(caption2) == len(caption)
    assert caption2.supervisions[0].text == caption.supervisions[0].text
    assert caption2.supervisions[1].text == caption.supervisions[1].text
    print("✅ test_round_trip_stringio passed")


def test_format_conversion_in_memory():
    """Test format conversion using in-memory data."""
    srt_content = """1
00:00:00,000 --> 00:00:02,000
Hello world
"""
    # Read SRT from string
    caption = Caption.from_string(srt_content, format="srt")

    # Convert to VTT using to_string
    vtt_content = caption.to_string(format="vtt")

    assert "WEBVTT" in vtt_content
    assert "Hello world" in vtt_content
    print("✅ test_format_conversion_in_memory passed")


def test_unicode_in_memory():
    """Test Unicode content with in-memory data."""
    unicode_content = """1
00:00:00,000 --> 00:00:02,000
你好世界

2
00:00:02,500 --> 00:00:05,000
日本語テスト 🎬
"""
    caption = Caption.from_string(unicode_content, format="srt")

    assert len(caption) == 2
    assert "你好世界" in caption.supervisions[0].text
    assert "日本語テスト" in caption.supervisions[1].text

    # Test round-trip
    data = caption.to_bytes()
    caption2 = Caption.read(io.BytesIO(data), format="srt")

    assert "你好世界" in caption2.supervisions[0].text
    print("✅ test_unicode_in_memory passed")


def test_speaker_labels_in_memory():
    """Test speaker labels with in-memory data."""
    txt_content = """Alice: Hello there
Bob: Hi, how are you?
Alice: I'm doing great!
"""
    caption = Caption.from_string(txt_content, format="txt")

    # Check speakers were parsed (if supported by txt parser)
    assert len(caption) == 3

    # Write back
    data = caption.to_bytes()
    assert isinstance(data, bytes)
    print("✅ test_speaker_labels_in_memory passed")


if __name__ == "__main__":
    test_read_from_stringio()
    test_read_from_bytesio()
    test_from_string()
    test_write_to_bytesio()
    test_to_bytes()
    test_round_trip_stringio()
    test_format_conversion_in_memory()
    test_unicode_in_memory()
    test_speaker_labels_in_memory()

    print("\n" + "=" * 50)
    print("🎉 All in-memory I/O tests passed!")
    print("=" * 50)
