"""Авторазметка GroundingDINO для расширения датасета."""

from .grounding import GroundingLabeler, GroundingLabelStats, label_images

__all__ = ["GroundingLabelStats", "GroundingLabeler", "label_images"]
