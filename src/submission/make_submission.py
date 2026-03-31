"""Submission helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.utils.misc import ensure_dir


def build_prediction_dataframe(
    sample_ids: Iterable[str],
    pred_indices: Iterable[int],
    id_column: str = "image_name",
    target_column: str = "pred_label",
    idx_to_label: Optional[Dict[int, str]] = None,
    use_label_name: bool = False,
) -> pd.DataFrame:
    """Create a submission dataframe from IDs and predicted class indices."""
    sample_ids = list(sample_ids)
    pred_indices = list(pred_indices)

    if use_label_name and idx_to_label is not None:
        mapped_values = [idx_to_label[int(i)] for i in pred_indices]
        if all(str(v).isdigit() for v in mapped_values):
            values: List[str | int] = [int(v) for v in mapped_values]
        else:
            values = mapped_values
    else:
        values = [int(i) for i in pred_indices]

    return pd.DataFrame({id_column: sample_ids, target_column: values})


def save_prediction_csv(
    submission_df: pd.DataFrame,
    output_dir: Path | str,
    filename: str = "prediction.csv",
) -> Path:
    """Save submission CSV with required filename."""
    output_dir = ensure_dir(output_dir)
    output_path = output_dir / filename
    submission_df.to_csv(output_path, index=False)
    return output_path
