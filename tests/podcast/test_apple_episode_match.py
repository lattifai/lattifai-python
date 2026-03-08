"""Tests for Apple episode matching in podcast RSS."""

import xml.etree.ElementTree as ET

from lattifai.podcast.client import PodcastLoader
from lattifai.podcast.types import PodcastMetadata


def test_find_apple_episode_matches_id_in_link_without_lookup() -> None:
    rss = ET.fromstring(
        """
        <rss>
          <channel>
            <item>
              <title>Episode A</title>
              <link>https://podcasts.apple.com/us/podcast/foo/id12345?i=1000749366733</link>
              <guid>guid-1</guid>
              <enclosure url="https://example.com/a.mp3" />
            </item>
            <item>
              <title>Episode B</title>
              <link>https://podcasts.apple.com/us/podcast/foo/id12345?i=1000749000000</link>
              <guid>guid-2</guid>
              <enclosure url="https://example.com/b.mp3" />
            </item>
          </channel>
        </rss>
        """
    )

    loader = PodcastLoader()
    podcast = PodcastMetadata(title="Test Podcast")

    episode = loader._find_apple_episode(rss, "1000749366733", podcast)

    assert episode.title == "Episode A"
    assert episode.audio_url == "https://example.com/a.mp3"
