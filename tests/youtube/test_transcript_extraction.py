"""Tests for YouTube external transcript URL extraction and parsing"""

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
    """Test _parse_transcript_html static method — outputs podcast-transcript Markdown"""

    YT_URL = "https://www.youtube.com/watch?v=TEST123"

    def test_timestamped_to_podcast_markdown(self):
        """Lex Fridman style → podcast-transcript format with YouTube links"""
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
        # Should produce podcast-transcript blocks
        assert "Lex Fridman\n" in text
        assert "[(00:00:00)](https://www.youtube.com/watch?t=0)" in text
        assert "Welcome to the show." in text
        assert "Guest\n" in text
        assert "[(00:00:05)](https://www.youtube.com/watch?t=5)" in text

    def test_timestamped_without_youtube_url(self):
        """Timestamped format without youtube_url uses anchor links"""
        html = """
        <html><body>
        <p>Speaker (00:01:30) Some text here.</p>
        </body></html>
        """
        text = YouTubeDownloader._parse_transcript_html(html)
        assert text is not None
        assert "[(00:01:30)](#90)" in text

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
        """Substack/Dwarkesh style → speaker + text blocks (no timestamps)"""
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
        <h2>Links</h2>
        <p>Subscribe to the podcast</p>
        <footer><p>Copyright 2025</p></footer>
        </body></html>
        """
        text = YouTubeDownloader._parse_transcript_html(html)
        assert text is not None
        # Each segment: Speaker\nText\n(blank line)
        assert "Dwarkesh Patel\n" in text
        assert "Dario Amodei\n" in text
        assert "So you have been saying AI could transform science." in text
        # Should not contain non-transcript content
        assert "Subscribe" not in text
        assert "Copyright" not in text
        assert "Share" not in text
        assert "Jun 26" not in text

    def test_substack_mixed_content_filters_correctly(self):
        """Dialogue extraction should ignore non-dialogue lines"""
        html = """
        <html><body>
        <p>Episode 42 of the podcast.</p>
        <p><strong>Host Name:</strong> Welcome to the show. Today we have a special guest.</p>
        <p><strong>Guest Person:</strong> Thanks for having me on the program.</p>
        <p><strong>Host Name:</strong> Let us dive right into the topic of AI safety.</p>
        <p>Thanks for listening!</p>
        </body></html>
        """
        text = YouTubeDownloader._parse_transcript_html(html)
        assert text is not None
        assert "Host Name\n" in text
        assert "Guest Person\n" in text
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
