"""Podcast client for metadata extraction and audio download."""

import json
import logging
import re
import tempfile
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    import defusedxml.ElementTree as SafeET
except ImportError:
    SafeET = None

import xml.etree.ElementTree as ET

from .platforms import detect_platform, extract_ids, resolve_rss_feed, validate_url
from .types import EpisodeMetadata, PodcastMetadata, PodcastPlatform

logger = logging.getLogger(__name__)

# Common User-Agent for HTTP requests
_USER_AGENT = "LattifAI/1.0 (Podcast Transcription)"

# iTunes / content namespace used in podcast RSS feeds
_ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
_CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"

# Response size limits
_MAX_RSS_SIZE = 50 * 1024 * 1024  # 50 MB for RSS feeds
_MAX_HTML_SIZE = 10 * 1024 * 1024  # 10 MB for HTML pages
_MAX_JSON_SIZE = 5 * 1024 * 1024  # 5 MB for JSON API responses

# Regex patterns for extracting guest names from episode titles
# NOTE: No re.IGNORECASE — require capitalized proper names (English)
_GUEST_PATTERNS = [
    re.compile(r"\b[Ww]ith\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"),
    re.compile(r"\b[Ff]t\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"),
    re.compile(r"\b[Ff]eaturing\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"),
    re.compile(r"[|:]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*$"),
    # Chinese patterns
    re.compile(r"(?:嘉宾|对话|专访|访谈)[：:]\s*(.+?)(?:\s*[|｜]|$)"),
]


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities from text."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _http_get(url: str, timeout: int = 20, max_size: int = _MAX_RSS_SIZE) -> bytes:
    """Perform an HTTP GET request with URL validation and size limits."""
    validate_url(url)
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        # Check Content-Length if available
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > max_size:
            raise ValueError(f"Response too large: {content_length} bytes (max {max_size})")
        data = resp.read(max_size + 1)
        if len(data) > max_size:
            raise ValueError(f"Response exceeded max size of {max_size} bytes")
        return data


def _http_get_json(url: str, timeout: int = 15) -> dict:
    """Perform an HTTP GET and parse JSON response."""
    data = _http_get(url, timeout=timeout, max_size=_MAX_JSON_SIZE)
    return json.loads(data.decode("utf-8"))


def _http_download_to_file(url: str, output_path: Path, timeout: int = 300) -> int:
    """Stream-download a file to disk without loading entirely into memory."""
    validate_url(url)
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    total = 0
    with urlopen(req, timeout=timeout) as resp, open(output_path, "wb") as f:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    return total


def _parse_xml(data: bytes) -> ET.Element:
    """Parse XML safely, using defusedxml if available."""
    if SafeET is not None:
        return SafeET.fromstring(data)
    logger.warning("defusedxml not installed; parsing RSS with stdlib ElementTree (less safe)")
    return ET.fromstring(data)


def _build_show_notes(author: str, guest_names: List[str], title: str, description: str) -> str:
    """Build show notes string from metadata components."""
    parts = []
    if author:
        parts.append(f"Host: {author}")
    if guest_names:
        parts.append(f"Guest(s): {', '.join(guest_names)}")
    if title:
        parts.append(f"Episode: {title}")
    if description:
        parts.append(f"Description: {description[:2000]}")
    return "\n".join(parts)


class PodcastLoader:
    """Podcast metadata extractor and audio downloader.

    Supports:
    - Apple Podcasts (via iTunes Lookup API + RSS)
    - RSS feeds (direct)
    - Xiaoyuzhou / 小宇宙 (via page scraping or RSS)
    - Spotify (metadata only, audio from original RSS for externally-hosted shows)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize PodcastLoader.

        Args:
            cache_dir: Directory for caching downloaded RSS feeds. If None, uses temp dir.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "lattifai_podcast"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rss_cache: Dict[str, ET.Element] = {}

    def get_episode_metadata(
        self,
        url: str,
        rss_feed_url: Optional[str] = None,
    ) -> EpisodeMetadata:
        """Extract episode metadata from a podcast URL.

        Args:
            url: Podcast episode URL (Apple Podcasts, RSS, Xiaoyuzhou, etc.)
            rss_feed_url: Override RSS feed URL (skips auto-detection).

        Returns:
            EpisodeMetadata with audio URL, show notes, speaker info.

        Raises:
            ValueError: If URL platform cannot be detected or metadata extraction fails.
        """
        platform = detect_platform(url)
        podcast_id, episode_id = extract_ids(url, platform)

        logger.info(f"Detected platform: {platform.value}, podcast_id={podcast_id}, episode_id={episode_id}")

        if platform == PodcastPlatform.YOUTUBE:
            raise ValueError(
                "YouTube URLs should be handled by the youtube() method. "
                "Use client.youtube() instead of client.podcast() for YouTube content."
            )

        if platform == PodcastPlatform.XIAOYUZHOU:
            return self._get_xiaoyuzhou_metadata(url, episode_id)

        # For Apple, RSS, Spotify - resolve RSS feed
        feed_url = rss_feed_url or resolve_rss_feed(url, platform)
        if not feed_url:
            if platform == PodcastPlatform.SPOTIFY:
                raise ValueError(
                    "Spotify episode detected. For Spotify-exclusive podcasts, "
                    "please provide the audio file directly. For externally-hosted "
                    "podcasts, provide the RSS feed URL via podcast.rss_feed_url parameter."
                )
            raise ValueError(f"Could not resolve RSS feed for URL: {url}")

        # Parse RSS and find episode
        rss_root = self._fetch_rss(feed_url)
        podcast_meta = self._parse_podcast_metadata(rss_root, feed_url, platform, podcast_id)

        if episode_id and platform == PodcastPlatform.APPLE:
            episode = self._find_apple_episode(rss_root, episode_id, podcast_meta)
        else:
            # For direct RSS links, try to match by URL or return latest
            episode = self._find_episode_from_rss(rss_root, url, podcast_meta)

        return episode

    def download_audio(
        self,
        episode: EpisodeMetadata,
        output_dir: Optional[str] = None,
    ) -> str:
        """Download episode audio to local file (streaming, memory-safe).

        Args:
            episode: Episode metadata containing audio_url.
            output_dir: Output directory. If None, uses cache_dir.

        Returns:
            Path to downloaded audio file.

        Raises:
            ValueError: If episode has no audio URL.
            RuntimeError: If download fails.
        """
        if not episode.audio_url:
            raise ValueError("Episode has no audio URL. Cannot download.")

        out_dir = Path(output_dir) if output_dir else self.cache_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename from URL
        parsed = urlparse(episode.audio_url)
        filename = Path(parsed.path).name or "episode.mp3"
        # Sanitize filename
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
        output_path = out_dir / filename

        if output_path.exists():
            logger.info(f"Audio file already exists: {output_path}")
            return str(output_path)

        logger.info(f"Downloading audio from: {episode.audio_url}")
        try:
            total_bytes = _http_download_to_file(episode.audio_url, output_path, timeout=300)
            logger.info(f"Audio downloaded: {output_path} ({total_bytes / 1024 / 1024:.1f} MB)")
            return str(output_path)
        except Exception as e:
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Failed to download audio from {episode.audio_url}: {e}")

    def _fetch_rss(self, feed_url: str) -> ET.Element:
        """Fetch and parse RSS feed XML (with XXE protection)."""
        if feed_url in self._rss_cache:
            return self._rss_cache[feed_url]

        logger.info(f"Fetching RSS feed: {feed_url}")
        data = _http_get(feed_url, timeout=30, max_size=_MAX_RSS_SIZE)
        root = _parse_xml(data)
        self._rss_cache[feed_url] = root
        return root

    def _parse_podcast_metadata(
        self,
        rss_root: ET.Element,
        feed_url: str,
        platform: PodcastPlatform,
        podcast_id: str = "",
    ) -> PodcastMetadata:
        """Extract podcast-level metadata from RSS XML."""
        channel = rss_root.find("channel")
        if channel is None:
            channel = rss_root

        title = _text(channel, "title")
        author = _text(channel, f"{{{_ITUNES_NS}}}author") or _text(channel, "author")
        description = _strip_html(_text(channel, "description") or _text(channel, f"{{{_ITUNES_NS}}}summary"))
        language = _text(channel, "language")
        image_url = ""
        image_el = channel.find(f"{{{_ITUNES_NS}}}image")
        if image_el is not None:
            image_url = image_el.get("href", "")

        categories = []
        for cat_el in channel.findall(f"{{{_ITUNES_NS}}}category"):
            cat_text = cat_el.get("text", "")
            if cat_text:
                categories.append(cat_text)

        return PodcastMetadata(
            title=title,
            author=author,
            description=description,
            rss_feed_url=feed_url,
            platform=platform,
            podcast_id=podcast_id,
            image_url=image_url,
            language=language,
            categories=categories,
        )

    def _find_apple_episode(
        self,
        rss_root: ET.Element,
        apple_episode_id: str,
        podcast: PodcastMetadata,
    ) -> EpisodeMetadata:
        """Find a specific episode in RSS by Apple episode ID.

        Apple episode IDs don't directly map to RSS GUIDs. We use the iTunes
        Lookup API to get the episode title, then match by title in the RSS.
        """
        try:
            lookup_url = f"https://itunes.apple.com/lookup?id={apple_episode_id}&entity=podcastEpisode"
            data = _http_get_json(lookup_url)
            results = data.get("results", [])
            if results:
                ep_data = results[0]
                target_title = ep_data.get("trackName", "")
                if target_title:
                    channel = rss_root.find("channel") or rss_root
                    for item in channel.findall("item"):
                        item_title = _text(item, "title")
                        if item_title and _titles_match(item_title, target_title):
                            return self._parse_rss_episode(item, podcast)
        except Exception as e:
            logger.warning(f"iTunes episode lookup failed: {e}")

        logger.warning(f"Could not find Apple episode {apple_episode_id}, returning latest episode")
        return self._get_latest_episode(rss_root, podcast)

    def _find_episode_from_rss(
        self,
        rss_root: ET.Element,
        url: str,
        podcast: PodcastMetadata,
    ) -> EpisodeMetadata:
        """Find episode from RSS feed by URL match or return latest."""
        channel = rss_root.find("channel") or rss_root
        normalized_url = _normalize_url(url)

        for item in channel.findall("item"):
            link = _text(item, "link")
            guid = _text(item, "guid")
            if (link and _normalize_url(link) == normalized_url) or (guid and guid == url):
                return self._parse_rss_episode(item, podcast)

        return self._get_latest_episode(rss_root, podcast)

    def _get_latest_episode(
        self,
        rss_root: ET.Element,
        podcast: PodcastMetadata,
    ) -> EpisodeMetadata:
        """Get the most recent episode from RSS feed."""
        channel = rss_root.find("channel") or rss_root
        items = channel.findall("item")
        if not items:
            raise ValueError("RSS feed contains no episodes")
        return self._parse_rss_episode(items[0], podcast)

    def _parse_rss_episode(
        self,
        item: ET.Element,
        podcast: PodcastMetadata,
    ) -> EpisodeMetadata:
        """Parse a single RSS <item> element into EpisodeMetadata."""
        title = _text(item, "title")
        description = _strip_html(
            _text(item, "description")
            or _text(item, f"{{{_ITUNES_NS}}}summary")
            or _text(item, f"{{{_CONTENT_NS}}}encoded")
        )

        # Audio URL from <enclosure>
        audio_url = ""
        enclosure = item.find("enclosure")
        if enclosure is not None:
            audio_url = enclosure.get("url", "")

        # Duration
        duration = 0.0
        dur_text = _text(item, f"{{{_ITUNES_NS}}}duration")
        if dur_text:
            duration = _parse_duration(dur_text)

        pub_date = _text(item, "pubDate")
        episode_id = _text(item, "guid") or ""

        # Extract speaker names from metadata
        host_names = [podcast.author] if podcast.author else []
        guest_names = []
        for pattern in _GUEST_PATTERNS:
            match = pattern.search(title)
            if match:
                guest_name = match.group(1).strip()
                if guest_name and guest_name not in guest_names:
                    guest_names.append(guest_name)

        return EpisodeMetadata(
            title=title,
            description=description,
            audio_url=audio_url,
            duration=duration,
            publish_date=pub_date,
            episode_id=episode_id,
            podcast=podcast,
            host_names=host_names,
            guest_names=guest_names,
            show_notes=_build_show_notes(podcast.author, guest_names, title, description),
        )

    def _get_xiaoyuzhou_metadata(self, url: str, episode_id: str) -> EpisodeMetadata:
        """Extract metadata from Xiaoyuzhou (小宇宙) episode page.

        Xiaoyuzhou embeds episode data in #__NEXT_DATA__ JSON script tag.
        Falls back to og: meta tags if __NEXT_DATA__ is not available.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for Xiaoyuzhou support. " "Install with: pip install beautifulsoup4"
            )

        logger.info(f"Fetching Xiaoyuzhou episode page: {url}")
        html_bytes = _http_get(url, timeout=20, max_size=_MAX_HTML_SIZE)
        html = html_bytes.decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")

        next_data_script = soup.find("script", id="__NEXT_DATA__")
        if next_data_script and next_data_script.string:
            return self._parse_xiaoyuzhou_next_data(next_data_script.string, url)

        return self._parse_xiaoyuzhou_og_tags(soup, url)

    def _parse_xiaoyuzhou_next_data(self, json_text: str, url: str) -> EpisodeMetadata:
        """Parse Xiaoyuzhou __NEXT_DATA__ JSON for episode metadata."""
        data = json.loads(json_text)
        props = data.get("props", {}).get("pageProps", {})
        episode_data = props.get("episode", props)

        title = episode_data.get("title", "")
        description = _strip_html(episode_data.get("description", "") or episode_data.get("shownotes", ""))

        # Audio URL
        audio_url = ""
        enclosure = episode_data.get("enclosure", {})
        if isinstance(enclosure, dict):
            audio_url = enclosure.get("url", "")
        elif isinstance(enclosure, str):
            audio_url = enclosure
        if not audio_url:
            media_key = episode_data.get("mediaKey", "")
            if media_key:
                audio_url = f"https://media.xyzcdn.net/{media_key}"

        duration = max(0.0, float(episode_data.get("duration", 0)))

        podcast_data = episode_data.get("podcast", {})
        podcast_author = podcast_data.get("author", "")

        podcast_meta = PodcastMetadata(
            title=podcast_data.get("title", ""),
            author=podcast_author,
            description=_strip_html(podcast_data.get("description", "")),
            platform=PodcastPlatform.XIAOYUZHOU,
            podcast_id=podcast_data.get("pid", ""),
        )

        guest_names = []
        for pattern in _GUEST_PATTERNS:
            match = pattern.search(title)
            if match:
                guest_name = match.group(1).strip()
                if guest_name:
                    guest_names.append(guest_name)

        host_names = [podcast_author] if podcast_author else []

        return EpisodeMetadata(
            title=title,
            description=description,
            audio_url=audio_url,
            duration=duration,
            publish_date=episode_data.get("pubDate", ""),
            episode_id=episode_data.get("eid", ""),
            podcast=podcast_meta,
            host_names=host_names,
            guest_names=guest_names,
            show_notes=_build_show_notes(podcast_author, guest_names, title, description),
        )

    def _parse_xiaoyuzhou_og_tags(self, soup, url: str) -> EpisodeMetadata:
        """Fallback: parse Xiaoyuzhou page using og: meta tags."""

        def _og(prop: str) -> str:
            tag = soup.find("meta", property=prop)
            return tag.get("content", "") if tag else ""

        title = _og("og:title")
        description = _og("og:description")
        audio_url = _og("og:audio") or ""

        return EpisodeMetadata(
            title=title,
            description=description,
            audio_url=audio_url,
            show_notes=_build_show_notes("", [], title, description),
        )


# --- Helper functions ---


def _text(element: ET.Element, tag: str) -> str:
    """Safely extract text from an XML element's child tag."""
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return ""


def _parse_duration(text: str) -> float:
    """Parse iTunes duration string (seconds, MM:SS, or HH:MM:SS) to float seconds."""
    text = text.strip()
    if not text:
        return 0.0

    # Pure number = seconds
    try:
        return max(0.0, float(text))
    except ValueError:
        pass

    # HH:MM:SS or MM:SS
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return max(0.0, int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
        elif len(parts) == 2:
            return max(0.0, int(parts[0]) * 60 + float(parts[1]))
    except (ValueError, IndexError):
        pass

    return 0.0


def _titles_match(title_a: str, title_b: str) -> bool:
    """Fuzzy title matching: normalize and compare (CJK-safe)."""

    def _norm(t: str) -> str:
        # Keep Unicode word chars (\w includes CJK), remove punctuation/whitespace
        return re.sub(r"[\s\W]+", "", t.lower())

    return _norm(title_a) == _norm(title_b)


def _normalize_url(url: str) -> str:
    """Normalize URL for comparison (strip fragment, trailing slash)."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}{'?' + parsed.query if parsed.query else ''}"
