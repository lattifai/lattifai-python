"""Demo: Segmented alignment for long audio files.

This example demonstrates how to use segmented alignment for processing
long audio files (e.g., 2-10 hours) efficiently.

Three segmentation strategies are available:
1. 'time': Split by fixed time intervals with overlap
2. 'caption': Split based on existing caption boundaries and gaps
3. 'adaptive': Hybrid approach - respect caption boundaries while limiting duration
"""

from lattifai import LattifAI
from lattifai.config import AlignmentConfig, CaptionConfig

# Example 1: Time-based segmentation (fixed 5-minute segments)
# Good for: Podcasts, lectures, long-form content without natural breaks
print("=" * 60)
print("Example 1: Time-based segmentation")
print("=" * 60)

caption_config_time = CaptionConfig(
    input_format="srt",
    output_format="srt",
    segment_strategy="time",  # Fixed time intervals
    segment_duration=300.0,  # 5 minutes per segment
    segment_overlap=2.0,  # 2 second overlap
)

alignment_config = AlignmentConfig(
    verbose=True,  # Show segmentation info
)

client = LattifAI(
    caption_config=caption_config_time,
    alignment_config=alignment_config,
)

# Align long audio with time-based segmentation
# aligned = client.alignment(
#     input_media="long_podcast.mp3",  # 2 hour podcast
#     input_caption="long_podcast.srt",
#     output_caption_path="long_podcast_aligned.srt",
# )

print("\nTime-based segmentation: Splits audio into 5-minute chunks with 2s overlap")
print("✓ Predictable memory usage")
print("✓ Good for uniform content")
print("✗ May split in middle of sentences\n")


# Example 2: Caption-based segmentation (natural breaks)
# Good for: Interviews, conversations with pauses, structured content
print("=" * 60)
print("Example 2: Caption-based segmentation")
print("=" * 60)

caption_config_caption = CaptionConfig(
    input_format="srt",
    output_format="srt",
    segment_strategy="caption",  # Based on caption boundaries
    segment_max_gap=5.0,  # Split on 5+ second gaps
    segment_overlap=2.0,  # 2 second overlap
)

client = LattifAI(
    caption_config=caption_config_caption,
    alignment_config=alignment_config,
)

# Align long audio with caption-based segmentation
# aligned = client.alignment(
#     input_media="interview.mp4",  # 90 minute interview
#     input_caption="interview.srt",
#     output_caption_path="interview_aligned.srt",
# )

print("\nCaption-based segmentation: Splits on natural breaks (5+ second gaps)")
print("✓ Preserves semantic boundaries")
print("✓ Good for conversations with pauses")
print("✗ Variable segment lengths\n")


# Example 3: Adaptive segmentation (best of both worlds)
# Good for: Most use cases - combines time limits with natural breaks
print("=" * 60)
print("Example 3: Adaptive segmentation (recommended)")
print("=" * 60)

caption_config_adaptive = CaptionConfig(
    input_format="srt",
    output_format="srt",
    segment_strategy="adaptive",  # Hybrid approach
    segment_duration=600.0,  # Max 10 minutes per segment
    segment_max_gap=5.0,  # Prefer splitting on 5+ second gaps
    segment_overlap=2.0,  # 2 second overlap
)

client = LattifAI(
    caption_config=caption_config_adaptive,
    alignment_config=alignment_config,
)

# Align long audio with adaptive segmentation
# aligned = client.alignment(
#     input_media="documentary.mp4",  # 3 hour documentary
#     input_caption="documentary.srt",
#     output_caption_path="documentary_aligned.srt",
# )

print("\nAdaptive segmentation: Respects natural breaks within 10-minute limit")
print("✓ Best balance of memory usage and quality")
print("✓ Handles various content types")
print("✓ Recommended for most use cases\n")


# Example 4: No segmentation (standard mode)
# Good for: Short audio (<30 minutes)
print("=" * 60)
print("Example 4: No segmentation (standard mode)")
print("=" * 60)

caption_config_standard = CaptionConfig(
    input_format="srt",
    output_format="srt",
    segment_strategy="none",  # Process entire audio at once
)

client = LattifAI(
    caption_config=caption_config_standard,
    alignment_config=alignment_config,
)

# Align short audio without segmentation
# aligned = client.alignment(
#     input_media="short_video.mp4",  # 10 minute video
#     input_caption="short_video.srt",
#     output_caption_path="short_video_aligned.srt",
# )

print("\nNo segmentation: Process entire audio at once")
print("✓ Fastest for short audio")
print("✓ Best quality for <30 min content")
print("✗ High memory usage for long audio\n")


# Configuration recommendations
print("=" * 60)
print("Configuration Recommendations")
print("=" * 60)
print(
    """
Audio Duration | Strategy  | segment_duration | segment_overlap
-----------------------------------------------------------------------------
< 30 min       | none      | N/A              | N/A
30 min - 2 hr  | adaptive  | 600.0 (10 min)   | 2.0
2 hr - 5 hr    | adaptive  | 300.0 (5 min)    | 2.0
> 5 hr         | time      | 180.0 (3 min)    | 2.0

segment_max_gap guidelines:
- Interviews/conversations: 5.0 seconds (default)
- Podcasts with music breaks: 10.0 seconds
- Audiobooks: 3.0 seconds
- Documentaries: 7.0 seconds

segment_overlap guidelines:
- Standard: 2.0 seconds (default)
- High precision: 3.0-5.0 seconds
- Fast processing: 1.0 second
"""
)
