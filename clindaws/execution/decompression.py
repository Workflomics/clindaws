"""Decompression policies for optimized-candidate output ports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping


def _consumer_fset_key(
    cf: Mapping[str, frozenset[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        sorted((str(dim), tuple(sorted(values))) for dim, values in cf.items())
    )


def _producer_fset_key(
    pf: Mapping[str, frozenset[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        sorted((str(dim), tuple(sorted(values))) for dim, values in pf.items())
    )


def _consumers_diverge(
    *,
    producer_fset: Mapping[str, frozenset[str]],
    consumer_fsets: tuple[Mapping[str, frozenset[str]], ...],
) -> bool:
    for dim, available in producer_fset.items():
        if not available:
            continue
        intersection = available
        for cf in consumer_fsets:
            req = cf.get(dim)
            if req is None:
                continue
            intersection = intersection & req
            if not intersection:
                return True
    return False


def _consumer_inputs_by_port(
    relevant_records: Iterable[dict[str, object]],
) -> dict[tuple[str, int], Mapping[str, frozenset[str]]]:
    consumer_input_fset_by_port: dict[
        tuple[str, int], Mapping[str, frozenset[str]]
    ] = {}
    for record in relevant_records:
        cand = str(record["candidate_id"])
        for input_port in tuple(record["input_ports"]):
            consumer_input_fset_by_port[(cand, int(input_port["port_idx"]))] = (
                input_port["port_values_fset"]
            )
    return consumer_input_fset_by_port


def _bindable_pairs_by_producer(
    bindable_pairs: set[tuple[str, int, str, int]],
) -> dict[tuple[str, int], list[tuple[str, int]]]:
    by_producer: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for prod_cand, prod_port, cons_cand, cons_port in bindable_pairs:
        by_producer[(prod_cand, int(prod_port))].append(
            (cons_cand, int(cons_port))
        )
    return by_producer


def _output_port_consumers_diverge_check(
    output_port: Mapping[str, object],
    *,
    candidate_id: str,
    bindable_pairs_by_producer: Mapping[tuple[str, int], Iterable[tuple[str, int]]],
    consumer_input_fset_by_port: Mapping[tuple[str, int], Mapping[str, frozenset[str]]],
    relevant_candidates: frozenset[str] | None,
) -> bool:
    if int(output_port.get("multiplicity", 1)) <= 1:
        return False
    sources = tuple(output_port.get("source_port_indices", ()))
    if len(sources) <= 1:
        return False
    producer_fset: Mapping[str, frozenset[str]] = output_port["port_values_fset"]
    if all(len(v) <= 1 for v in producer_fset.values()):
        return False

    merged_idx = int(output_port["port_idx"])
    seen: set[tuple[str, int]] = set()
    consumer_fsets: list[Mapping[str, frozenset[str]]] = []
    for cons_cand, cons_port in bindable_pairs_by_producer.get(
        (candidate_id, merged_idx), ()
    ):
        if relevant_candidates is not None and cons_cand not in relevant_candidates:
            continue
        key = (cons_cand, cons_port)
        if key in seen:
            continue
        seen.add(key)
        cf = consumer_input_fset_by_port.get(key)
        if cf is not None:
            consumer_fsets.append(cf)
    if len(consumer_fsets) <= 1:
        return False
    return _consumers_diverge(
        producer_fset=producer_fset,
        consumer_fsets=tuple(consumer_fsets),
    )


def split_output_ports_kcluster(
    *,
    relevant_records: Iterable[dict[str, object]],
    bindable_pairs: set[tuple[str, int, str, int]],
    relevant_candidates: frozenset[str] | None,
) -> dict[str, int]:
    consumer_input_fset_by_port = _consumer_inputs_by_port(relevant_records)
    bindable_pairs_by_producer = _bindable_pairs_by_producer(bindable_pairs)
    stats = {
        "kcluster_split_ports_examined": 0,
        "kcluster_split_ports_diverged": 0,
        "kcluster_split_extra_ports_emitted": 0,
        "kcluster_split_classes_total": 0,
    }
    pairs_to_delete: list[tuple[str, int, str, int]] = []
    pairs_to_add: list[tuple[str, int, str, int]] = []

    for record in relevant_records:
        cand = str(record["candidate_id"])
        rebuilt: list[Mapping[str, object]] = []
        for output_port in tuple(record["output_ports"]):
            if int(output_port.get("multiplicity", 1)) <= 1:
                rebuilt.append(output_port)
                continue
            stats["kcluster_split_ports_examined"] += 1
            if not _output_port_consumers_diverge_check(
                output_port,
                candidate_id=cand,
                bindable_pairs_by_producer=bindable_pairs_by_producer,
                consumer_input_fset_by_port=consumer_input_fset_by_port,
                relevant_candidates=relevant_candidates,
            ):
                rebuilt.append(output_port)
                continue
            sources = tuple(int(s) for s in output_port["source_port_indices"])
            merged_idx = int(output_port["port_idx"])
            multiplicity = int(output_port["multiplicity"])
            producer_fset: Mapping[str, frozenset[str]] = output_port["port_values_fset"]
            consumers = tuple(
                (cons_cand, cons_port)
                for cons_cand, cons_port in bindable_pairs_by_producer.get(
                    (cand, merged_idx), ()
                )
                if relevant_candidates is None or cons_cand in relevant_candidates
            )

            class_members: list[list[tuple[str, int]]] = []
            class_intersections: list[dict[str, frozenset[str]]] = []
            for cons in consumers:
                cf = consumer_input_fset_by_port.get(cons)
                if cf is None:
                    continue
                placed = False
                for cls_idx, cls_intersection in enumerate(class_intersections):
                    candidate_intersection: dict[str, frozenset[str]] = {}
                    ok = True
                    for dim, current in cls_intersection.items():
                        req = cf.get(dim)
                        if req is None:
                            candidate_intersection[dim] = current
                            continue
                        merged = current & req
                        if not merged:
                            ok = False
                            break
                        candidate_intersection[dim] = merged
                    if ok:
                        class_members[cls_idx].append(cons)
                        class_intersections[cls_idx] = candidate_intersection
                        placed = True
                        break
                if not placed:
                    seed = {
                        dim: (available & cf[dim]) if dim in cf else available
                        for dim, available in producer_fset.items()
                    }
                    class_members.append([cons])
                    class_intersections.append(seed)

            num_classes = len(class_members)
            if num_classes <= 1 or num_classes > len(sources):
                rebuilt.append(output_port)
                continue

            sub_port_indices = sources[:num_classes]
            base = multiplicity // num_classes
            extra = multiplicity % num_classes
            multiplicities = [
                base + (1 if i < extra else 0) for i in range(num_classes)
            ]
            for sub_idx, mult in zip(sub_port_indices, multiplicities):
                split = dict(output_port)
                split["port_idx"] = sub_idx
                split["multiplicity"] = mult
                split["source_port_indices"] = (sub_idx,)
                rebuilt.append(split)

            for cons_cand, cons_port in bindable_pairs_by_producer.get(
                (cand, merged_idx), ()
            ):
                pairs_to_delete.append((cand, merged_idx, cons_cand, cons_port))
            for cons in consumers:
                cf = consumer_input_fset_by_port.get(cons)
                if cf is None:
                    continue
                for cls_idx, cls_intersection in enumerate(class_intersections):
                    compatible = True
                    for dim, current in cls_intersection.items():
                        req = cf.get(dim)
                        if req is None:
                            continue
                        if not (current & req):
                            compatible = False
                            break
                    if compatible:
                        pairs_to_add.append(
                            (cand, sub_port_indices[cls_idx], cons[0], cons[1])
                        )

            stats["kcluster_split_ports_diverged"] += 1
            stats["kcluster_split_extra_ports_emitted"] += num_classes - 1
            stats["kcluster_split_classes_total"] += num_classes
        record["output_ports"] = tuple(rebuilt)

    for entry in pairs_to_delete:
        bindable_pairs.discard(entry)
    for entry in pairs_to_add:
        bindable_pairs.add(entry)

    return stats


def split_output_ports_one_to_n(
    *,
    relevant_records: Iterable[dict[str, object]],
    bindable_pairs: set[tuple[str, int, str, int]],
    relevant_candidates: frozenset[str] | None,
    min_step_by_candidate: Mapping[str, int],
    max_step_by_candidate: Mapping[str, int],
) -> dict[str, int]:
    consumer_input_fset_by_port = _consumer_inputs_by_port(relevant_records)
    bindable_pairs_by_producer = _bindable_pairs_by_producer(bindable_pairs)
    stats = {
        "one_to_n_split_ports_examined": 0,
        "one_to_n_split_ports_diverged": 0,
        "one_to_n_split_extra_ports_emitted": 0,
        "one_to_n_split_cache_hits": 0,
        "one_to_n_split_cache_misses": 0,
        "one_to_n_split_consumers_dropped_temporal": 0,
    }
    pairs_to_delete: list[tuple[str, int, str, int]] = []
    pairs_to_add: list[tuple[str, int, str, int]] = []
    divergence_cache: dict[
        tuple[
            tuple[tuple[str, tuple[str, ...]], ...],
            tuple[tuple[tuple[str, tuple[str, ...]], ...], ...],
        ],
        bool,
    ] = {}

    for record in relevant_records:
        cand = str(record["candidate_id"])
        producer_min_step = min_step_by_candidate.get(cand)
        rebuilt: list[Mapping[str, object]] = []
        for output_port in tuple(record["output_ports"]):
            if int(output_port.get("multiplicity", 1)) <= 1:
                rebuilt.append(output_port)
                continue
            sources = tuple(int(s) for s in output_port.get("source_port_indices", ()))
            if len(sources) <= 1:
                rebuilt.append(output_port)
                continue
            producer_fset: Mapping[str, frozenset[str]] = output_port["port_values_fset"]
            if all(len(v) <= 1 for v in producer_fset.values()):
                rebuilt.append(output_port)
                continue

            stats["one_to_n_split_ports_examined"] += 1
            merged_idx = int(output_port["port_idx"])
            consumers_list: list[tuple[tuple[str, int], Mapping[str, frozenset[str]]]] = []
            for cons_cand, cons_port in bindable_pairs_by_producer.get(
                (cand, merged_idx), ()
            ):
                if relevant_candidates is not None and cons_cand not in relevant_candidates:
                    continue
                cons_max_step = max_step_by_candidate.get(cons_cand)
                if (
                    producer_min_step is not None
                    and cons_max_step is not None
                    and producer_min_step >= cons_max_step
                ):
                    stats["one_to_n_split_consumers_dropped_temporal"] += 1
                    continue
                cf = consumer_input_fset_by_port.get((cons_cand, cons_port))
                if cf is not None:
                    consumers_list.append(((cons_cand, cons_port), cf))

            if len(consumers_list) <= 1:
                rebuilt.append(output_port)
                continue

            consumer_fsets_tuple = tuple(cf for _, cf in consumers_list)
            cache_key = (
                _producer_fset_key(producer_fset),
                tuple(sorted(_consumer_fset_key(cf) for cf in consumer_fsets_tuple)),
            )
            diverges = divergence_cache.get(cache_key)
            if diverges is None:
                stats["one_to_n_split_cache_misses"] += 1
                diverges = _consumers_diverge(
                    producer_fset=producer_fset,
                    consumer_fsets=consumer_fsets_tuple,
                )
                divergence_cache[cache_key] = diverges
            else:
                stats["one_to_n_split_cache_hits"] += 1

            if not diverges:
                rebuilt.append(output_port)
                continue

            for sub_idx in sources:
                split = dict(output_port)
                split["port_idx"] = sub_idx
                split["multiplicity"] = 1
                split["source_port_indices"] = (sub_idx,)
                rebuilt.append(split)

            sym_pairs = record.setdefault("sym_output_pairs", [])
            for prev_idx, next_idx in zip(sources, sources[1:]):
                sym_pairs.append((next_idx, prev_idx))

            for cons_cand, cons_port in bindable_pairs_by_producer.get(
                (cand, merged_idx), ()
            ):
                pairs_to_delete.append((cand, merged_idx, cons_cand, cons_port))
            for (cons_cand, cons_port), _cf in consumers_list:
                for sub_idx in sources:
                    pairs_to_add.append((cand, sub_idx, cons_cand, cons_port))

            stats["one_to_n_split_ports_diverged"] += 1
            stats["one_to_n_split_extra_ports_emitted"] += len(sources) - 1
        record["output_ports"] = tuple(rebuilt)

    for entry in pairs_to_delete:
        bindable_pairs.discard(entry)
    for entry in pairs_to_add:
        bindable_pairs.add(entry)

    return stats
