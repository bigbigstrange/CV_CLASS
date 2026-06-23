from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DefectRecord:
    """Unified sample record for KSDD and KSDD2."""

    dataset: str
    split: str
    sample_id: str
    image_path: Path
    mask_path: Path | None
    has_mask: bool
    is_defect: bool
    is_segmented: bool = True

    @property
    def group_id(self) -> str:
        """Physical item id used for leak-free splitting (KSDD kosXX)."""
        if self.dataset == "ksdd":
            return self.sample_id.split("/")[0]
        return self.sample_id
