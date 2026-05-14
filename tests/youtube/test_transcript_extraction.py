"""Tests for YouTube external transcript URL extraction and parsing"""

import socket
from unittest.mock import MagicMock, patch

import pytest

from lattifai.youtube.client import YouTubeDownloader


class TestExtractTranscriptUrlFromDescription:
    """Test _extract_transcript_url_from_description static method"""

    def test_lex_fridman_format(self):
        """Lex Fridman style: *Transcript:* on one line, URL on next line"""
        desc = (
            "Some intro text.\n"
            "\n"
            "*Transcript:*\n"
            "https://lexfridman.com/peter-steinberger-transcript\n"
            "\n"
            "*CONTACT LEX:*\n"
            "https://lexfridman.com/survey\n"
        )
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://lexfridman.com/peter-steinberger-transcript"

    def test_inline_transcript_colon(self):
        """Inline format: Transcript: URL"""
        desc = "Transcript: https://example.com/my-transcript"
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://example.com/my-transcript"

    def test_inline_transcript_dash(self):
        """Inline format: Transcript - URL"""
        desc = "Transcript - https://example.com/my-transcript"
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://example.com/my-transcript"

    def test_markdown_bold_transcript(self):
        """Markdown bold: **Transcript:**"""
        desc = "**Transcript:**\n" "https://example.com/transcript\n"
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://example.com/transcript"

    def test_underscore_italic_transcript(self):
        """Markdown italic: _Transcript:_"""
        desc = "_Transcript:_\n" "https://example.com/transcript\n"
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://example.com/transcript"

    def test_transcript_with_blank_line_before_url(self):
        """Transcript label followed by blank line then URL"""
        desc = "*Transcript:*\n" "\n" "https://example.com/transcript\n"
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://example.com/transcript"

    def test_no_transcript(self):
        """No transcript URL in description"""
        desc = "Just a regular video about cooking with no transcript link."
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url is None

    def test_empty_description(self):
        """Empty description"""
        url = YouTubeDownloader._extract_transcript_url_from_description("")
        assert url is None

    def test_none_description(self):
        """None description"""
        url = YouTubeDownloader._extract_transcript_url_from_description(None)
        assert url is None

    def test_transcript_word_in_unrelated_context(self):
        """Word 'transcript' appears but not as a label for a URL"""
        desc = "We provide a transcript of the episode on our website."
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url is None

    def test_latent_space_show_notes_inline(self):
        """Latent Space style: Substack Article w/Show Notes: URL"""
        desc = (
            "Jeff Dean joins us to discuss AI.\n"
            "\n"
            "Substack Article w/Show Notes: https://www.latent.space/p/jeffdean\n"
            "\n"
            "Timestamps:\n"
        )
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://www.latent.space/p/jeffdean"

    def test_show_notes_standalone_label(self):
        """Show Notes as standalone label with URL on next line"""
        desc = "*Show Notes:*\n" "https://example.com/episode-42-notes\n"
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://example.com/episode-42-notes"

    def test_show_notes_colon(self):
        """Inline: Show Notes: URL"""
        desc = "Show Notes: https://example.com/notes"
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://example.com/notes"

    def test_show_notes_dash(self):
        """Inline: Show Notes - URL"""
        desc = "Show Notes - https://example.com/notes"
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://example.com/notes"

    def test_dwarkesh_bullet_transcript(self):
        """Dwarkesh Patel style: * Transcript: URL (bullet list item)"""
        desc = (
            "Dario Amodei thinks we are just a few years away.\n"
            "\n"
            "𝐄𝐏𝐈𝐒𝐎𝐃𝐄 𝐋𝐈𝐍𝐊𝐒\n"
            "* Transcript: https://www.dwarkesh.com/p/dario-amodei-2\n"
            "* Apple Podcasts: https://podcasts.apple.com/us/podcast/xxx\n"
            "* Spotify: https://open.spotify.com/episode/xxx\n"
        )
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://www.dwarkesh.com/p/dario-amodei-2"

    def test_bullet_dash_transcript(self):
        """Bullet with dash: - Transcript: URL"""
        desc = "- Transcript: https://example.com/ep1-transcript"
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url == "https://example.com/ep1-transcript"

    def test_url_not_mistaken_for_transcript(self):
        """A non-transcript URL should not be extracted"""
        desc = "Check out our website:\n" "https://example.com\n" "\n" "Subscribe for more!\n"
        url = YouTubeDownloader._extract_transcript_url_from_description(desc)
        assert url is None


class TestParseTranscriptHtml:
    """Test _parse_transcript_html static method — outputs markdown transcript format"""

    YT_URL = "https://www.youtube.com/watch?v=TEST123"

    def test_timestamped_to_markdown(self):
        """Lex Fridman style → markdown format: **Speaker:** text [HH:MM:SS]"""
        html = """
        <html><body>
        <div class="entry-content">
            <p>Lex Fridman (00:00:00) Welcome to the show.</p>
            <p>Guest (00:00:05) Thanks for having me.</p>
            <p>Lex Fridman (00:00:10) Let's get started.</p>
        </div>
        </body></html>
        """
        text = YouTubeDownloader._parse_transcript_html(html, youtube_url=self.YT_URL)
        assert text is not None
        # Should produce markdown format: **Speaker:** text [HH:MM:SS]
        assert "**Lex Fridman:**" in text
        assert "Welcome to the show." in text
        assert "[00:00:00]" in text
        assert "**Guest:**" in text
        assert "[00:00:05]" in text

    def test_timestamped_without_youtube_url(self):
        """Timestamped format without youtube_url still includes timestamp"""
        html = """
        <html><body>
        <p>Speaker (00:01:30) Some text here.</p>
        </body></html>
        """
        text = YouTubeDownloader._parse_transcript_html(html)
        assert text is not None
        assert "**Speaker:**" in text
        assert "[00:01:30]" in text

    def test_skips_script_and_style(self):
        """Script and style tags should be excluded"""
        html = """
        <html><body>
        <script>var x = 'should not appear';</script>
        <style>.hidden { display: none; }</style>
        <p>Speaker (00:00:00) Actual content here.</p>
        </body></html>
        """
        text = YouTubeDownloader._parse_transcript_html(html)
        assert text is not None
        assert "should not appear" not in text
        assert "hidden" not in text
        assert "Actual content here" in text

    def test_skips_nav_and_footer(self):
        """Navigation and footer content should be excluded"""
        html = """
        <html><body>
        <nav><a href="/">Home</a></nav>
        <p>Speaker (00:00:00) Content.</p>
        <footer>Copyright 2024</footer>
        </body></html>
        """
        text = YouTubeDownloader._parse_transcript_html(html)
        assert text is not None
        assert "Home" not in text
        assert "Copyright" not in text
        assert "Content" in text

    def test_substack_dialogue_to_markdown(self):
        """Substack/Dwarkesh style → markdown format (no timestamps, >=10 segments needed)"""
        html = """
        <html><body>
        <nav><a href="/">Home</a></nav>
        <h1>Dario Amodei Interview</h1>
        <p>Jun 26, 2025</p>
        <p>A conversation about AI scaling.</p>
        <div class="button-wrapper"><a href="#">Share</a><a href="#">Like</a></div>
        <h2>Transcript</h2>
        <p><strong>Dwarkesh Patel:</strong> So you have been saying AI could transform science.</p>
        <p><strong>Dario Amodei:</strong> Yes, and I think it is important to be specific about this.</p>
        <p><strong>Dwarkesh Patel:</strong> That is a huge claim. What gives you confidence?</p>
        <p><strong>Dario Amodei:</strong> The key insight is that AI systems are getting better at reasoning.</p>
        <p><strong>Dwarkesh Patel:</strong> Can you give a concrete example of that?</p>
        <p><strong>Dario Amodei:</strong> Sure, look at how models handle multi-step problems now.</p>
        <p><strong>Dwarkesh Patel:</strong> And how does that compare to a year ago?</p>
        <p><strong>Dario Amodei:</strong> The improvement has been dramatic in both speed and accuracy.</p>
        <p><strong>Dwarkesh Patel:</strong> What about the safety implications of these advances?</p>
        <p><strong>Dario Amodei:</strong> Safety is core to everything we do at Anthropic.</p>
        <p><strong>Dwarkesh Patel:</strong> Thanks for sharing your perspective on all of this.</p>
        <h2>Links</h2>
        <p>Subscribe to the podcast</p>
        <footer><p>Copyright 2025</p></footer>
        </body></html>
        """
        text = YouTubeDownloader._parse_transcript_html(html)
        assert text is not None
        # Each segment: **Speaker:** text
        assert "**Dwarkesh Patel:**" in text
        assert "**Dario Amodei:**" in text
        assert "So you have been saying AI could transform science." in text
        # Should not contain non-transcript content
        assert "Subscribe" not in text
        assert "Copyright" not in text
        assert "Share" not in text
        assert "Jun 26" not in text

    def test_substack_mixed_content_filters_correctly(self):
        """Dialogue extraction should ignore non-dialogue lines (>=10 segments needed)"""
        html = """
        <html><body>
        <p>Episode 42 of the podcast.</p>
        <p><strong>Host Name:</strong> Welcome to the show. Today we have a special guest.</p>
        <p><strong>Guest Person:</strong> Thanks for having me on the program.</p>
        <p><strong>Host Name:</strong> Let us dive right into the topic of AI safety.</p>
        <p><strong>Guest Person:</strong> AI safety is incredibly important for the future.</p>
        <p><strong>Host Name:</strong> Can you elaborate on the specific risks you see?</p>
        <p><strong>Guest Person:</strong> The main risk is misalignment between goals and actions.</p>
        <p><strong>Host Name:</strong> And how do we address that in practice?</p>
        <p><strong>Guest Person:</strong> Through careful training and evaluation procedures.</p>
        <p><strong>Host Name:</strong> What about the role of regulation in all of this?</p>
        <p><strong>Guest Person:</strong> Regulation needs to be thoughtful and well-informed.</p>
        <p><strong>Host Name:</strong> Thank you for sharing your insights with us today.</p>
        <p>Thanks for listening!</p>
        </body></html>
        """
        text = YouTubeDownloader._parse_transcript_html(html)
        assert text is not None
        assert "**Host Name:**" in text
        assert "**Guest Person:**" in text
        assert "Episode 42" not in text
        assert "Thanks for listening" not in text

    def test_empty_html(self):
        """Empty HTML returns None"""
        text = YouTubeDownloader._parse_transcript_html("<html><body></body></html>")
        assert text is None

    def test_no_body(self):
        """HTML without body tag"""
        text = YouTubeDownloader._parse_transcript_html("<html><head></head></html>")
        assert text is None

    def test_standalone_speaker_names(self):
        """Strategy 3b: standalone speaker names on their own block (Dwarkesh article format)"""
        # Build a transcript with 2 speakers appearing ≥3 times each, with chapter headings
        blocks = [
            "Some intro",
            "Transcript",
            "00:00:00 – Opening",
        ]
        # Add enough speaker+text blocks to trigger Strategy 3b (≥10 segments)
        for i in range(6):
            blocks.append("Alice Smith")
            blocks.append(f"This is what Alice says in segment {i}. She talks about many interesting things.")
            blocks.append("Bob Jones")
            blocks.append(f"And Bob responds with his thoughts on segment {i}. He has a different perspective.")

        text = "\n\n".join(blocks)
        transcript = YouTubeDownloader._parse_transcript_html(text)
        assert transcript is not None
        assert "**Alice Smith:**" in transcript
        assert "**Bob Jones:**" in transcript
        assert "## Opening" in transcript
        assert "[00:00:00]" in transcript
        # Should have ≥12 speaker segments
        import re

        speaker_lines = [l for l in transcript.split("\n") if l.startswith("**")]
        assert len(speaker_lines) >= 12

    def test_standalone_speaker_skips_short_blocks(self):
        """Strategy 3b skips blocks shorter than 20 chars as noise"""
        blocks = ["Transcript"]
        for i in range(6):
            blocks.append("Speaker One")
            blocks.append(f"A full paragraph of meaningful dialogue content number {i}.")
            blocks.append("103")  # UI noise (like count)
            blocks.append("Speaker Two")
            blocks.append(f"Another paragraph of response content for segment {i} here.")

        text = "\n\n".join(blocks)
        transcript = YouTubeDownloader._parse_transcript_html(text)
        assert transcript is not None
        assert "103" not in transcript

    def test_standalone_speaker_needs_minimum_occurrences(self):
        """Strategy 3b requires ≥3 occurrences and ≥2 distinct speakers"""
        # Only 1 occurrence of each — should NOT trigger Strategy 3b
        blocks = [
            "Transcript",
            "Alice",
            "Hello there, this is a sufficiently long paragraph for testing.",
            "Bob",
            "Hi Alice, nice to meet you today for this conversation.",
        ]
        text = "\n\n".join(blocks)
        transcript = YouTubeDownloader._parse_transcript_html(text)
        # Should fallback to raw text, not produce **Alice:** format
        assert transcript is None or "**Alice:**" not in transcript


class TestIsHijackedPage:
    """Test _is_hijacked_page static method"""

    def test_normal_html(self):
        assert not YouTubeDownloader._is_hijacked_page("<html><body><p>Hello</p></body></html>")

    def test_ssl_error_page(self):
        html = "<html><body>" + "x" * 100 + "<h1>NET::ERR_CERT_COMMON_NAME_INVALID</h1></body></html>"
        assert YouTubeDownloader._is_hijacked_page(html)

    def test_chinese_privacy_warning(self):
        html = "<html><body>" + "x" * 100 + "您的连接不是私密连接</body></html>"
        assert YouTubeDownloader._is_hijacked_page(html)

    def test_pem_certificate(self):
        html = "x" * 100 + "-----BEGIN CERTIFICATE-----\nMIIBxx..."
        assert YouTubeDownloader._is_hijacked_page(html)

    def test_empty_html(self):
        assert not YouTubeDownloader._is_hijacked_page("")
        assert not YouTubeDownloader._is_hijacked_page(None)


class TestIsHostReachable:
    """Test _is_host_reachable static method (mocked: never touches real network).

    The previous version of these tests made real TCP/DNS calls (google.com,
    192.0.2.1, .invalid) which is flaky behind captive portals, transparent
    proxies, ISP DNS hijacking, or offline CI runners. We mock socket directly
    to assert the function's contract instead.
    """

    def test_returns_true_when_connection_succeeds(self):
        mock_sock = MagicMock()
        with patch("socket.create_connection", return_value=mock_sock) as mock_conn:
            assert YouTubeDownloader._is_host_reachable("example.com", port=443, timeout=5)
        mock_conn.assert_called_once_with(("example.com", 443), timeout=5)

    def test_returns_false_on_timeout(self):
        with patch("socket.create_connection", side_effect=socket.timeout):
            assert not YouTubeDownloader._is_host_reachable("slow.example.com", timeout=1)

    def test_returns_false_on_connection_refused(self):
        with patch("socket.create_connection", side_effect=ConnectionRefusedError):
            assert not YouTubeDownloader._is_host_reachable("127.0.0.1", port=1, timeout=1)

    def test_returns_false_on_dns_failure(self):
        with patch("socket.create_connection", side_effect=socket.gaierror):
            assert not YouTubeDownloader._is_host_reachable("nonexistent.invalid", timeout=1)


class TestRepairJson:
    """Test _repair_json and parse_json_response from llm.base"""

    def test_trailing_comma(self):
        from lattifai.llm.base import parse_json_response

        result = parse_json_response('[{"a": 1}, {"b": 2},]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_missing_comma_between_objects(self):
        from lattifai.llm.base import parse_json_response

        result = parse_json_response('[{"a": 1} {"b": 2}]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_truncated_array(self):
        from lattifai.llm.base import parse_json_response

        result = parse_json_response('[{"a": 1}, {"b": 2}')
        assert result == [{"a": 1}, {"b": 2}]

    def test_valid_json_untouched(self):
        from lattifai.llm.base import parse_json_response

        result = parse_json_response('[{"a": 1}, {"b": 2}]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_code_fence_stripped(self):
        from lattifai.llm.base import parse_json_response

        result = parse_json_response('```json\n[{"a": 1}]\n```')
        assert result == [{"a": 1}]

    def test_thinking_tokens_stripped(self):
        from lattifai.llm.base import parse_json_response

        result = parse_json_response('<think>reasoning</think>[{"a": 1}]')
        assert result == [{"a": 1}]
