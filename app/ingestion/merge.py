from typing import Any, Dict, List, Optional, Sequence, Tuple

import polars as pl


def _column_sets(frames: Sequence[pl.DataFrame]) -> List[set]:
    return [set(df.columns) for df in frames]


def assess_compatibility(
    named_frames: Sequence[Tuple[str, pl.DataFrame]],
) -> Dict[str, Any]:
    """Decide whether files can be stacked, joined, or must stay independent.

    Never recommends a silent concat of mismatched schemas.
    """
    names = [n for n, _ in named_frames]
    frames = [df for _, df in named_frames]
    if len(frames) < 2:
        cols = list(frames[0].columns) if frames else []
        return {
            "mode": "single",
            "can_concat": True,
            "can_join": False,
            "shared_columns": cols,
            "union_columns": cols,
            "message": "Only one file loaded.",
            "files": names,
        }

    sets = _column_sets(frames)
    shared = set.intersection(*sets)
    union = set.union(*sets)
    identical = all(s == sets[0] for s in sets)

    if identical:
        return {
            "mode": "concat",
            "can_concat": True,
            "can_join": bool(shared),
            "shared_columns": sorted(shared),
            "union_columns": sorted(union),
            "message": (
                "All files share the same columns. You can stack them into one table "
                "(row-wise union). A join is optional if you need a key-based merge instead."
            ),
            "files": names,
        }

    if shared:
        return {
            "mode": "ambiguous",
            "can_concat": False,
            "can_join": True,
            "shared_columns": sorted(shared),
            "union_columns": sorted(union),
            "message": (
                "Schemas differ. NEXUS will not stack these files automatically. "
                "Join on shared columns, stack with nulls for missing columns (explicit), "
                "or load one file only."
            ),
            "files": names,
        }

    return {
        "mode": "independent",
        "can_concat": False,
        "can_join": False,
        "shared_columns": [],
        "union_columns": sorted(union),
        "message": (
            "No shared columns — these look like unrelated datasets. "
            "Choose one file to load. They will not be combined."
        ),
        "files": names,
    }


def concat_frames(
    frames: Sequence[pl.DataFrame],
    how: str = "vertical",
) -> pl.DataFrame:
    """Stack frames. `vertical` requires identical columns; `diagonal` allows extra nulls."""
    if not frames:
        raise ValueError("No frames to concatenate.")
    if len(frames) == 1:
        return frames[0]
    if how == "vertical":
        return pl.concat(list(frames), how="vertical")
    return pl.concat(list(frames), how="diagonal")


def join_frames(
    left: pl.DataFrame,
    right: pl.DataFrame,
    keys: Sequence[str],
    how: str = "inner",
) -> pl.DataFrame:
    if not keys:
        raise ValueError("Join requires at least one key column.")
    missing_left = [k for k in keys if k not in left.columns]
    missing_right = [k for k in keys if k not in right.columns]
    if missing_left or missing_right:
        raise ValueError(
            f"Join keys missing. Left missing {missing_left}; right missing {missing_right}."
        )
    how_map = {"inner": "inner", "left": "left", "full": "full"}
    join_how = how_map.get(how, "inner")
    return left.join(right, on=list(keys), how=join_how)  # type: ignore[arg-type]


def successive_join(
    frames: Sequence[pl.DataFrame],
    keys: Sequence[str],
    how: str = "inner",
) -> pl.DataFrame:
    merged = frames[0]
    for nxt in frames[1:]:
        merged = join_frames(merged, nxt, keys, how=how)
    return merged
