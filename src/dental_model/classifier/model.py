"""
src/dental_model/classifier/model.py
Dental lesion severity classifier wrapping a timm backbone.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


class DentalClassifier(nn.Module):
    """
    Dental lesion severity classifier wrapping a timm backbone.

    Attributes
    ----------
    backbone_name : str
        Name of the timm backbone (e.g., 'efficientnet_b0', 'convnext_tiny').
    num_classes : int
        Number of output prediction classes.
    pretrained : bool
        Whether to load pre-trained ImageNet weights.
    drop_rate : float
        Dropout probability applied to the classifier head.
    model : nn.Module
        The underlying timm model instance.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.drop_rate = dropout

        logger.info(
            "Initializing DentalClassifier with backbone='%s', num_classes=%d, "
            "pretrained=%s, dropout=%.2f",
            backbone,
            num_classes,
            pretrained,
            dropout,
        )

        self.model = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            drop_rate=self.drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass on batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        return self.model(x)

    @classmethod
    def from_config(cls, config_path_or_dict: str | Path | dict[str, Any]) -> DentalClassifier:
        """
        Construct a DentalClassifier instance from a YAML config file or dictionary.

        Parameters
        ----------
        config_path_or_dict : str, Path, or dict
            Path to classifier YAML config or loaded configuration dictionary.

        Returns
        -------
        DentalClassifier
            Configured classifier model.
        """
        if isinstance(config_path_or_dict, (str, Path)):
            path = Path(config_path_or_dict)
            if not path.is_absolute():
                path = REPO_ROOT / path
            if not path.exists():
                raise FileNotFoundError(f"Classifier config not found: {path}")
            with path.open(encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
        else:
            cfg = config_path_or_dict

        backbone = cfg.get("model", {}).get("backbone", "efficientnet_b0")
        pretrained = cfg.get("model", {}).get("pretrained", True)
        dropout = cfg.get("model", {}).get("dropout", 0.2)
        classes = cfg.get("data", {}).get("classes", ["healthy", "caries_early", "caries_advanced"])
        num_classes = len(classes)

        return cls(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
