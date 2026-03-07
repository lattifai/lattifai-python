"""Podcast platform detection and RSS feed resolution."""

import ipaddress
import json
import logging
import re
import socket
from typing import Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .types import PodcastPlatform

logger = logging.getLogger(__name__)


def validate_url(url: str) -> None:
    """Validate URL scheme and host to prevent SSRF attacks.

    Only allows http/https schemes and blocks requests to internal/private networks.
    Resolves hostnames to IP addresses and checks all forms of loopback, private,
    link-local, and reserved addresses.

    Raises:
        ValueError: If URL scheme is not http/https or host is internal/private.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}. Only http and https are allowed.")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("URL has no hostname.")

    # Block well-known local hostnames
    if hostname in ("localhost", "localhost.localdomain"):
        raise ValueError(f"Blocked URL to internal host: {hostname}")

    # Resolve hostname and check all resolved IPs
    try:
        addr_infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        # If we can't resolve, allow — the fetch itself will fail
        return

    for info in addr_infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"Blocked URL resolving to non-public address: {hostname} -> {ip_str}")


# URL patterns for platform detection
_APPLE_PATTERN = re.compile(r"podcasts\.apple\.com/.+/podcast/.+/id(\d+)")
_APPLE_EPISODE_PATTERN = re.compile(r"podcasts\.apple\.com/.+/podcast/.+/id(\d+)\?i=(\d+)")
_SPOTIFY_SHOW_PATTERN = re.compile(r"open\.spotify\.com/show/([a-zA-Z0-9]+)")
_SPOTIFY_EPISODE_PATTERN = re.compile(r"open\.spotify\.com/episode/([a-zA-Z0-9]+)")
_XIAOYUZHOU_EPISODE_PATTERN = re.compile(r"(?:www\.)?xiaoyuzhoufm\.com/episode/([a-f0-9]+)")
_XIAOYUZHOU_PODCAST_PATTERN = re.compile(r"(?:www\.)?xiaoyuzhoufm\.com/podcast/([a-f0-9]+)")
_YOUTUBE_PATTERN = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)")


def detect_platform(url: str) -> PodcastPlatform:
    """Detect podcast platform from URL.

    Args:
        url: Podcast URL to analyze.

    Returns:
        PodcastPlatform enum value.
    """
    if not url:
        return PodcastPlatform.UNKNOWN

    parsed = urlparse(url)
    host = parsed.hostname or ""

    if "podcasts.apple.com" in host:
        return PodcastPlatform.APPLE
    if "open.spotify.com" in host:
        return PodcastPlatform.SPOTIFY
    if "xiaoyuzhoufm.com" in host:
        return PodcastPlatform.XIAOYUZHOU
    if "youtube.com" in host or "youtu.be" in host:
        return PodcastPlatform.YOUTUBE

    # Check if URL points to an RSS/XML feed
    path = parsed.path.lower()
    if path.endswith((".xml", ".rss", "/feed", "/rss")):
        return PodcastPlatform.RSS

    # Try content-type detection for ambiguous URLs
    if any(kw in url.lower() for kw in ("feed", "rss", "xml")):
        return PodcastPlatform.RSS

    return PodcastPlatform.UNKNOWN


def extract_ids(url: str, platform: PodcastPlatform) -> Tuple[str, str]:
    """Extract podcast and episode IDs from URL.

    Args:
        url: Platform URL.
        platform: Detected platform.

    Returns:
        Tuple of (podcast_id, episode_id). Either may be empty string if not found.
    """
    if platform == PodcastPlatform.APPLE:
        ep_match = _APPLE_EPISODE_PATTERN.search(url)
        if ep_match:
            return ep_match.group(1), ep_match.group(2)
        show_match = _APPLE_PATTERN.search(url)
        if show_match:
            return show_match.group(1), ""

    elif platform == PodcastPlatform.SPOTIFY:
        ep_match = _SPOTIFY_EPISODE_PATTERN.search(url)
        if ep_match:
            return "", ep_match.group(1)
        show_match = _SPOTIFY_SHOW_PATTERN.search(url)
        if show_match:
            return show_match.group(1), ""

    elif platform == PodcastPlatform.XIAOYUZHOU:
        ep_match = _XIAOYUZHOU_EPISODE_PATTERN.search(url)
        if ep_match:
            return "", ep_match.group(1)
        pod_match = _XIAOYUZHOU_PODCAST_PATTERN.search(url)
        if pod_match:
            return pod_match.group(1), ""

    elif platform == PodcastPlatform.YOUTUBE:
        yt_match = _YOUTUBE_PATTERN.search(url)
        if yt_match:
            return "", yt_match.group(1)

    return "", ""


def resolve_rss_feed(url: str, platform: PodcastPlatform) -> Optional[str]:
    """Resolve RSS feed URL from a platform-specific podcast URL.

    Args:
        url: Platform podcast URL.
        platform: Detected platform.

    Returns:
        RSS feed URL string, or None if resolution fails.
    """
    if platform == PodcastPlatform.RSS:
        return url

    if platform == PodcastPlatform.APPLE:
        return _resolve_apple_rss(url)

    if platform == PodcastPlatform.XIAOYUZHOU:
        return _resolve_xiaoyuzhou_rss(url)

    # Spotify and YouTube don't have direct RSS resolution
    return None


def _resolve_apple_rss(url: str) -> Optional[str]:
    """Resolve RSS feed from Apple Podcasts URL via iTunes Lookup API.

    Apple Podcasts URL contains a podcast ID. The iTunes Lookup API returns
    metadata including the feedUrl field pointing to the original RSS feed.
    """
    match = _APPLE_PATTERN.search(url)
    if not match:
        logger.warning(f"Could not extract podcast ID from Apple URL: {url}")
        return None

    podcast_id = match.group(1)
    lookup_url = f"https://itunes.apple.com/lookup?id={podcast_id}&entity=podcast"

    try:
        req = Request(lookup_url, headers={"User-Agent": "LattifAI/1.0"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = data.get("results", [])
        if results:
            feed_url = results[0].get("feedUrl")
            if feed_url:
                logger.info(f"Resolved Apple Podcasts RSS feed: {feed_url}")
                return feed_url

        logger.warning(f"No feedUrl found in iTunes Lookup for podcast ID {podcast_id}")
    except Exception as e:
        logger.warning(f"iTunes Lookup API failed for {podcast_id}: {e}")

    return None


def _resolve_xiaoyuzhou_rss(url: str) -> Optional[str]:
    """Resolve RSS feed for Xiaoyuzhou podcast.

    Xiaoyuzhou (小宇宙) podcasts may have RSS feeds at feed.xyzfm.space/<podcast_id>.
    Also supports RSSHub route: rsshub.app/xiaoyuzhou/podcast/<podcast_id>.
    """
    pod_match = _XIAOYUZHOU_PODCAST_PATTERN.search(url)
    ep_match = _XIAOYUZHOU_EPISODE_PATTERN.search(url)

    podcast_id = None
    if pod_match:
        podcast_id = pod_match.group(1)
    elif ep_match:
        # For episode URLs, we need to fetch the page to get podcast_id
        # This will be handled by the client
        pass

    if podcast_id:
        # Try official feed first
        feed_url = f"https://feed.xyzfm.space/{podcast_id}"
        try:
            req = Request(feed_url, headers={"User-Agent": "LattifAI/1.0"}, method="HEAD")
            with urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return feed_url
        except Exception:
            pass

        # Fallback to RSSHub
        rsshub_url = f"https://rsshub.app/xiaoyuzhou/podcast/{podcast_id}"
        logger.info(f"Trying RSSHub for Xiaoyuzhou podcast: {rsshub_url}")
        return rsshub_url

    return None
