from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Optional

from owlready2 import Thing, ThingClass, owl


@dataclass(frozen=True)
class ClassStats:
    depth: int | None
    sibling_count: int
    subclass_count: int
    parent_count: int


def _class_pool(classes: Optional[Iterable[ThingClass]]) -> set[ThingClass] | None:
    if classes is None:
        return None
    if isinstance(classes, set):
        return classes
    return {cls for cls in classes if isinstance(cls, ThingClass)}


def direct_parents(entity, classes: Optional[Iterable[ThingClass]] = None) -> list[ThingClass]:
    pool = _class_pool(classes)
    parents = [
        parent
        for parent in getattr(entity, "is_a", []) or []
        if isinstance(parent, ThingClass) and parent != Thing and parent != owl.Thing
    ]
    if pool is not None:
        parents = [parent for parent in parents if parent in pool]
    return parents


def direct_subclasses(entity, classes: Optional[Iterable[ThingClass]] = None) -> list[ThingClass]:
    if not isinstance(entity, ThingClass):
        return []
    pool = _class_pool(classes)
    try:
        children = [child for child in entity.subclasses() if isinstance(child, ThingClass)]
    except Exception:
        children = []
    if pool is not None:
        children = [child for child in children if child in pool]
    return children


def siblings(entity, classes: Optional[Iterable[ThingClass]] = None) -> set[ThingClass]:
    pool = _class_pool(classes)
    values: set[ThingClass] = set()
    for parent in direct_parents(entity, pool):
        values.update(direct_subclasses(parent, pool))
    values.discard(entity)
    return values


def class_depth(
    entity,
    classes: Optional[Iterable[ThingClass]] = None,
    memo: Optional[dict[ThingClass, int | None]] = None,
    visiting: Optional[set[ThingClass]] = None,
) -> int | None:
    if not isinstance(entity, ThingClass):
        return None
    if entity == Thing or entity == owl.Thing:
        return 0

    pool = _class_pool(classes)
    memo = memo if memo is not None else {}
    visiting = visiting if visiting is not None else set()
    if entity in memo:
        return memo[entity]
    if entity in visiting:
        return None

    visiting.add(entity)
    parent_depths = [
        depth
        for parent in direct_parents(entity, pool)
        for depth in [class_depth(parent, pool, memo, visiting)]
        if depth is not None and isfinite(depth)
    ]
    visiting.discard(entity)

    depth = max(parent_depths) + 1 if parent_depths else 1
    memo[entity] = depth
    return depth


def class_stats(entity, classes: Optional[Iterable[ThingClass]] = None) -> ClassStats:
    pool = _class_pool(classes)
    return ClassStats(
        depth=class_depth(entity, pool),
        sibling_count=len(siblings(entity, pool)),
        subclass_count=len(direct_subclasses(entity, pool)),
        parent_count=len(direct_parents(entity, pool)),
    )


def global_class_metrics(classes: Iterable[ThingClass]) -> dict[str, int]:
    class_list = [cls for cls in classes if isinstance(cls, ThingClass) and cls != Thing and cls != owl.Thing]
    stats = [class_stats(cls, class_list) for cls in class_list]
    return {
        "max_depth": max((stat.depth or 0 for stat in stats), default=1),
        "max_sibling_count": max((stat.sibling_count for stat in stats), default=1),
        "max_subclass_count": max((stat.subclass_count for stat in stats), default=1),
        "max_parent_count": max((stat.parent_count for stat in stats), default=1),
    }


def selection_weight(entity, metrics: dict[str, int], classes: Optional[Iterable[ThingClass]] = None) -> float:
    stats = class_stats(entity, classes)
    max_depth = metrics.get("max_depth") or 1
    max_siblings = metrics.get("max_sibling_count") or 1
    max_subclasses = metrics.get("max_subclass_count") or 1
    max_parents = metrics.get("max_parent_count") or 1

    normalized_depth = (stats.depth or 0) / max_depth
    normalized_siblings = stats.sibling_count / max_siblings
    normalized_subclasses = stats.subclass_count / max_subclasses
    normalized_parents = stats.parent_count / max_parents
    return normalized_depth * (normalized_siblings + 1) / (normalized_subclasses + normalized_parents + 1)
