"""
Caption Standardization Module

实现广播级字幕标准化，参考 Netflix/BBC 指南：
- 时间轴清理（最小/最大时长，间隔检查）
- 智能文本换行
- 质量验证

Reference Standards:
- Netflix Timed Text Style Guide
- BBC Subtitle Guidelines
- EBU-TT-D Standard
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

from lhotse.supervision import SupervisionSegment

from .supervision import Supervision

__all__ = [
    "CaptionStandardizer",
    "CaptionValidator",
    "StandardizationConfig",
    "ValidationResult",
]


@dataclass
class StandardizationConfig:
    """标准化配置"""

    min_duration: float = 0.8
    """最小持续时间（秒）- Netflix 推荐 5/6 秒，BBC 推荐 0.3 秒"""

    max_duration: float = 7.0
    """最大持续时间（秒）- Netflix/BBC 推荐 7 秒"""

    min_gap: float = 0.08
    """最小间隔（秒）- 防止字幕闪烁，80ms 是安全阈值"""

    max_lines: int = 2
    """最大行数 - 广播标准通常为 2 行"""

    max_chars_per_line: int = 42
    """每行最大字符数 - 英文 42，中文/日文 22"""

    optimal_cps: float = 17.0
    """最佳阅读速度（字符/秒）- Netflix 推荐 17-20 CPS"""

    def __post_init__(self):
        """验证配置参数"""
        if self.min_duration <= 0:
            raise ValueError("min_duration must be positive")
        if self.max_duration <= self.min_duration:
            raise ValueError("max_duration must be greater than min_duration")
        if self.min_gap < 0:
            raise ValueError("min_gap cannot be negative")
        if self.max_lines < 1:
            raise ValueError("max_lines must be at least 1")
        if self.max_chars_per_line < 10:
            raise ValueError("max_chars_per_line must be at least 10")


@dataclass
class ValidationResult:
    """验证结果"""

    valid: bool = True
    """是否通过所有验证"""

    warnings: List[str] = field(default_factory=list)
    """警告消息列表"""

    # 统计指标
    avg_cps: float = 0.0
    """平均阅读速度（字符/秒）"""

    max_cpl: int = 0
    """最大行字符数"""

    segments_too_short: int = 0
    """过短片段数"""

    segments_too_long: int = 0
    """过长片段数"""

    gaps_too_small: int = 0
    """间隔过小数"""


class CaptionStandardizer:
    """
    字幕标准化处理器

    处理流程:
    1. 时间轴清理 - 调整时长和间隔
    2. 文本格式化 - 智能换行
    3. 验证 - 生成质量指标

    Example:
        >>> standardizer = CaptionStandardizer(min_duration=0.8, max_chars_per_line=42)
        >>> processed = standardizer.process(supervisions)
    """

    # 中文标点（用于换行优先级）
    CJK_PUNCTUATION = r"[，。！？；、：""''（）【】《》…—]"

    # 英文标点
    EN_PUNCTUATION = r"[,.!?;:\-]"

    # 所有可分割标点
    ALL_PUNCTUATION = r"[，。！？；、：,.!?;:\s]"

    def __init__(
        self,
        min_duration: float = 0.8,
        max_duration: float = 7.0,
        min_gap: float = 0.08,
        max_lines: int = 2,
        max_chars_per_line: int = 42,
    ):
        """
        初始化标准化器

        Args:
            min_duration: 最小持续时间（秒）
            max_duration: 最大持续时间（秒）
            min_gap: 最小间隔（秒）
            max_lines: 最大行数
            max_chars_per_line: 每行最大字符数
        """
        self.config = StandardizationConfig(
            min_duration=min_duration,
            max_duration=max_duration,
            min_gap=min_gap,
            max_lines=max_lines,
            max_chars_per_line=max_chars_per_line,
        )

    def process(
        self, segments: List[Union[Supervision, SupervisionSegment]]
    ) -> List[Supervision]:
        """
        主处理入口

        Args:
            segments: 原始字幕片段列表

        Returns:
            处理后的字幕片段列表
        """
        if not segments:
            return []

        # 1. 按开始时间排序
        sorted_segments = sorted(segments, key=lambda s: s.start)

        # 2. 时间轴清理
        processed = self._sanitize_timeline(sorted_segments)

        # 3. 文本格式化
        processed = self._format_texts(processed)

        return processed

    def _sanitize_timeline(
        self, segments: List[Union[Supervision, SupervisionSegment]]
    ) -> List[Supervision]:
        """
        时间轴清理

        处理逻辑:
        A. 间隔检查 - 确保字幕间有足够间隔
        B. 最小时长检查 - 延长过短的字幕
        C. 最大时长检查 - 截断过长的字幕

        优先级策略: 间隔 > 最小时长（间隔不足会导致显示问题）
        """
        result: List[Supervision] = []

        for i, seg in enumerate(segments):
            # 创建新实例
            new_seg = self._copy_segment(seg)

            # A. 检查与前一个字幕的间隔
            if result:
                prev_seg = result[-1]
                prev_end = prev_seg.start + prev_seg.duration
                gap = new_seg.start - prev_end

                if gap < self.config.min_gap:
                    # 间隔太小或重叠
                    # 目标: prev_end_new + min_gap = new_seg.start
                    # => prev_duration_new = new_seg.start - min_gap - prev_seg.start
                    target_prev_duration = new_seg.start - self.config.min_gap - prev_seg.start

                    if target_prev_duration >= self.config.min_duration:
                        # 可以安全缩短前一个字幕（仍满足最小时长）
                        result[-1] = self._copy_segment(
                            prev_seg, duration=target_prev_duration
                        )
                    else:
                        # 缩短前一个字幕会让它低于最小时长，推迟当前字幕开始时间
                        new_start = prev_end + self.config.min_gap
                        duration_diff = new_start - seg.start
                        new_duration = max(
                            0.1,  # 保证至少有一点持续时间
                            new_seg.duration - duration_diff,
                        )
                        new_seg = self._copy_segment(
                            new_seg, start=new_start, duration=new_duration
                        )

            # B. 最小时长检查
            if new_seg.duration < self.config.min_duration:
                # 检查是否会与下一个字幕重叠
                next_start = (
                    segments[i + 1].start if i + 1 < len(segments) else float("inf")
                )
                max_extend = next_start - new_seg.start - self.config.min_gap
                new_duration = min(self.config.min_duration, max(max_extend, new_seg.duration))
                new_seg = self._copy_segment(new_seg, duration=new_duration)

            # C. 最大时长检查
            if new_seg.duration > self.config.max_duration:
                new_seg = self._copy_segment(new_seg, duration=self.config.max_duration)

            result.append(new_seg)

        return result

    def _format_texts(
        self, segments: List[Supervision]
    ) -> List[Supervision]:
        """对所有字幕应用文本格式化"""
        return [
            self._copy_segment(seg, text=self._smart_split_text(seg.text or ""))
            for seg in segments
        ]

    def _smart_split_text(self, text: str) -> str:
        """
        智能文本换行

        优先级:
        1. 中文标点位置（，。！？等）
        2. 英文标点位置（,.!?等）
        3. 空格位置
        4. 强制分割（硬截断）

        Args:
            text: 原始文本

        Returns:
            换行后的文本
        """
        # 清理文本
        text = self._normalize_text(text)

        # 检查是否需要换行
        if len(text) <= self.config.max_chars_per_line:
            return text

        lines: List[str] = []
        remaining = text

        for _ in range(self.config.max_lines):
            if len(remaining) <= self.config.max_chars_per_line:
                lines.append(remaining)
                remaining = ""
                break

            # 寻找最佳分割点
            split_pos = self._find_split_point(remaining, self.config.max_chars_per_line)

            lines.append(remaining[:split_pos].rstrip())
            remaining = remaining[split_pos:].lstrip()

        # 如果还有剩余文本且已达到最大行数，追加到最后一行
        if remaining and lines:
            # 可以选择截断或保留，这里选择追加（可能超出字符限制）
            lines[-1] = lines[-1] + " " + remaining if lines[-1] else remaining

        return "\n".join(lines)

    def _find_split_point(self, text: str, max_len: int) -> int:
        """
        寻找最佳分割点

        策略：在 max_len 附近寻找标点或空格
        搜索范围：40% - 110% 的 max_len

        Args:
            text: 待分割文本
            max_len: 最大长度

        Returns:
            分割位置索引
        """
        search_start = int(max_len * 0.4)
        search_end = min(len(text), int(max_len * 1.1))

        best_pos = max_len
        best_priority = 999  # 优先级越小越好

        # 从后向前搜索，优先找靠近 max_len 的分割点
        for i in range(min(search_end, len(text)) - 1, search_start - 1, -1):
            char = text[i]
            priority = self._get_split_priority(char)

            if priority < best_priority:
                best_priority = priority
                best_pos = i + 1  # 在标点/空格后分割

                # 如果找到最高优先级（中文标点），可以提前退出
                if priority == 1:
                    break

        return best_pos

    def _get_split_priority(self, char: str) -> int:
        """
        获取字符的分割优先级

        Returns:
            1 = 中文标点（最高优先级）
            2 = 英文标点
            3 = 空格
            999 = 其他字符（不适合分割）
        """
        if re.match(self.CJK_PUNCTUATION, char):
            return 1
        elif re.match(self.EN_PUNCTUATION, char):
            return 2
        elif char.isspace():
            return 3
        return 999

    def _normalize_text(self, text: str) -> str:
        """
        文本规范化

        - 移除多余空白
        - 移除已有换行符（将重新格式化）
        - 统一空格
        """
        # 移除已有换行符
        text = text.replace("\n", " ")
        # 合并多余空白
        text = re.sub(r"\s+", " ", text.strip())
        return text

    def _copy_segment(
        self,
        seg: Union[Supervision, SupervisionSegment],
        **overrides,
    ) -> Supervision:
        """
        创建 Supervision 的副本

        Args:
            seg: 原始片段
            **overrides: 要覆盖的字段

        Returns:
            新的 Supervision 实例
        """
        return Supervision(
            id=overrides.get("id", seg.id),
            recording_id=overrides.get("recording_id", seg.recording_id),
            start=overrides.get("start", seg.start),
            duration=overrides.get("duration", seg.duration),
            channel=overrides.get("channel", getattr(seg, "channel", None)),
            text=overrides.get("text", seg.text),
            language=overrides.get("language", getattr(seg, "language", None)),
            speaker=overrides.get("speaker", getattr(seg, "speaker", None)),
            gender=overrides.get("gender", getattr(seg, "gender", None)),
            custom=overrides.get("custom", getattr(seg, "custom", None)),
            alignment=overrides.get("alignment", getattr(seg, "alignment", None)),
        )


class CaptionValidator:
    """
    字幕质量验证器

    验证字幕是否符合广播标准，并生成质量指标报告。

    Example:
        >>> validator = CaptionValidator()
        >>> result = validator.validate(supervisions)
        >>> if not result.valid:
        ...     print(result.warnings)
    """

    def __init__(
        self,
        config: Optional[StandardizationConfig] = None,
        min_duration: float = 0.8,
        max_duration: float = 7.0,
        min_gap: float = 0.08,
        max_chars_per_line: int = 42,
    ):
        """
        初始化验证器

        Args:
            config: 标准化配置（如果提供，忽略其他参数）
            min_duration: 最小持续时间
            max_duration: 最大持续时间
            min_gap: 最小间隔
            max_chars_per_line: 每行最大字符数
        """
        if config:
            self.config = config
        else:
            self.config = StandardizationConfig(
                min_duration=min_duration,
                max_duration=max_duration,
                min_gap=min_gap,
                max_chars_per_line=max_chars_per_line,
            )

    def validate(
        self, segments: List[Union[Supervision, SupervisionSegment]]
    ) -> ValidationResult:
        """
        验证字幕并返回质量指标

        Args:
            segments: 字幕片段列表

        Returns:
            ValidationResult 包含验证结果和指标
        """
        result = ValidationResult()

        if not segments:
            return result

        total_cps = 0.0
        prev_end = 0.0

        for i, seg in enumerate(segments):
            text = seg.text or ""
            duration = seg.duration

            # CPS 计算（排除换行符）
            text_length = len(text.replace("\n", ""))
            cps = text_length / duration if duration > 0 else 0
            total_cps += cps

            # CPL 计算
            lines = text.split("\n")
            max_line_len = max((len(line) for line in lines), default=0)
            result.max_cpl = max(result.max_cpl, max_line_len)

            # 时长检查
            if duration < self.config.min_duration:
                result.segments_too_short += 1
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): duration {duration:.2f}s < min {self.config.min_duration}s"
                )

            if duration > self.config.max_duration:
                result.segments_too_long += 1
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): duration {duration:.2f}s > max {self.config.max_duration}s"
                )

            # 间隔检查
            if i > 0:
                gap = seg.start - prev_end
                if gap < self.config.min_gap and gap >= 0:
                    result.gaps_too_small += 1
                    result.warnings.append(
                        f"Segment {i} (id={seg.id}): gap {gap:.3f}s < min {self.config.min_gap}s"
                    )

            # CPL 检查
            if max_line_len > self.config.max_chars_per_line:
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): line length {max_line_len} > max {self.config.max_chars_per_line}"
                )

            # CPS 检查（过快的阅读速度）
            if cps > self.config.optimal_cps * 1.5:  # 超过最佳速度 50%
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): CPS {cps:.1f} exceeds recommended {self.config.optimal_cps}"
                )

            prev_end = seg.start + seg.duration

        # 计算平均 CPS
        result.avg_cps = total_cps / len(segments)

        # 判断是否通过验证
        result.valid = (
            result.segments_too_short == 0
            and result.segments_too_long == 0
            and result.gaps_too_small == 0
        )

        return result


def standardize_captions(
    segments: List[Union[Supervision, SupervisionSegment]],
    min_duration: float = 0.8,
    max_duration: float = 7.0,
    min_gap: float = 0.08,
    max_lines: int = 2,
    max_chars_per_line: int = 42,
) -> List[Supervision]:
    """
    便捷函数：标准化字幕列表

    Args:
        segments: 原始字幕片段列表
        min_duration: 最小持续时间（秒）
        max_duration: 最大持续时间（秒）
        min_gap: 最小间隔（秒）
        max_lines: 最大行数
        max_chars_per_line: 每行最大字符数

    Returns:
        处理后的字幕片段列表

    Example:
        >>> from lattifai.caption import standardize_captions
        >>> processed = standardize_captions(supervisions, max_chars_per_line=22)
    """
    standardizer = CaptionStandardizer(
        min_duration=min_duration,
        max_duration=max_duration,
        min_gap=min_gap,
        max_lines=max_lines,
        max_chars_per_line=max_chars_per_line,
    )
    return standardizer.process(segments)
