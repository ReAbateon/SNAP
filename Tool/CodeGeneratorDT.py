# 
# Copyright (c) 2026 Lorenzo Abate <lorenzo.abate@unina.it>.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import argparse
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Dict, Tuple
import csv
from pathlib import Path

import joblib


INT16_MIN = -32768
INT16_MAX = 32767
UINT16_MIN = 0
UINT16_MAX = 65535

def to_int16(x: float) -> int:
    # round-to-nearest + clamp
    v = int(round(x))
    if v < INT16_MIN:
        return INT16_MIN
    if v > INT16_MAX:
        return INT16_MAX
    return v

def to_uint16(x: int) -> int:
    if x < UINT16_MIN:
        return UINT16_MIN
    if x > UINT16_MAX:
        return UINT16_MAX
    return x

# ----------------------------
# Helpers: load + sklearn tree
# ----------------------------

def extract_model(obj: Any) -> Any:
    """Support a dict bundle with a 'model' key or a direct model object."""
    if isinstance(obj, dict):
        if "model" not in obj:
            raise KeyError("The bundle (dict) does not contain the 'model' key.")
        return obj["model"]
    return obj


def get_sklearn_tree(model: Any):
    if hasattr(model, "tree_"):
        return model.tree_
    raise TypeError("The extracted model does not look like an sklearn DecisionTree (missing .tree_).")


# ----------------------------
# Core: table generation
# ----------------------------

@dataclass(frozen=True)
class SubtreeTask:
    root_node: int
    prefix_bits: str  # global path bits used to reach this subtree


@dataclass
class TableResult:
    root_node: int
    prefix_bits: str
    V: int
    table: List[List[int]]         # len=2^V, ogni entry: node_ids lungo il path (len V+1)
    frontier_end: List[int]        # len=2^V, node_id alla frontiera (dopo V decisioni)


def build_table_from_node(tree, start_node: int, V: int) -> TableResult:
    """
    Build a 2^V table starting from start_node.

    Encoding (MSB-first in each chunk):
      - bit 1 => TRUE  => left
      - bit 0 => FALSE => right

    If a leaf is reached before depth V:
      - all suffix extensions of the current prefix are filled
      - the leaf id is replicated until the stored path has V+1 nodes
    """
    if V <= 0:
        raise ValueError("V must be > 0")

    n_entries = 1 << V
    table: List[Optional[List[int]]] = [None] * n_entries
    frontier_end: List[Optional[int]] = [None] * n_entries

    cl = tree.children_left
    cr = tree.children_right

    def is_leaf(n: int) -> bool:
        return cl[n] == -1 and cr[n] == -1

    def fill_all_suffixes(prefix_bits_int: int, depth: int, path_nodes: List[int]) -> None:
        remaining = V - depth
        base = prefix_bits_int << remaining
        n_suffix = 1 << remaining

        leaf_id = path_nodes[-1]
        # Keep path length equal to V+1 nodes (including root).
        full_path = path_nodes + [leaf_id] * (V + 1 - len(path_nodes))

        for suffix in range(n_suffix):
            idx = base | suffix
            if table[idx] is None:
                table[idx] = full_path
                frontier_end[idx] = leaf_id

    def dfs(node_id: int, depth: int, bits_int: int, path_nodes: List[int]) -> None:
        path_nodes.append(node_id)

        if depth == V:
            if table[bits_int] is None:
                table[bits_int] = path_nodes.copy()
                frontier_end[bits_int] = node_id
            path_nodes.pop()
            return

        if is_leaf(node_id):
            fill_all_suffixes(bits_int, depth, path_nodes)
            path_nodes.pop()
            return

        dfs(cl[node_id], depth + 1, (bits_int << 1) | 1, path_nodes)  # TRUE -> left -> 1
        dfs(cr[node_id], depth + 1, (bits_int << 1) | 0, path_nodes)  # FALSE -> right -> 0

        path_nodes.pop()

    dfs(start_node, 0, 0, [])

    missing = sum(1 for x in table if x is None)
    if missing != 0:
        raise RuntimeError(f"Incomplete table: {missing} missing entries out of {n_entries}")

    return TableResult(
        root_node=start_node,
        prefix_bits="",  # filled by the caller
        V=V,
        table=[x for x in table if x is not None],
        frontier_end=[x for x in frontier_end if x is not None],
    )


def generate_all_tables(joblib_path: str, V: int) -> Tuple[Any, List[TableResult]]:
    """
    Generate all chunked tables: one for the root and one for each internal node
    found on a chunk frontier.
    """
    loaded = joblib.load(joblib_path)
    model = extract_model(loaded)
    n_features = model.n_features_in_
    tree = get_sklearn_tree(model)
    max_depth = loaded["meta"]["max_depth"]
    #print("Max depth:", max_depth)

    cl = tree.children_left
    cr = tree.children_right

    def is_leaf(n: int) -> bool:
        return cl[n] == -1 and cr[n] == -1

    results: List[TableResult] = []
    visited: Set[int] = set()

    q = deque([SubtreeTask(root_node=0, prefix_bits="")])

    while q:
        task = q.popleft()
        if task.root_node in visited:
            continue
        visited.add(task.root_node)

        tr = build_table_from_node(tree, task.root_node, V)
        tr.prefix_bits = task.prefix_bits
        results.append(tr)

        for i, end_node in enumerate(tr.frontier_end):
            if not is_leaf(end_node):
                chunk_bits = format(i, f"0{V}b")
                new_prefix = task.prefix_bits + chunk_bits
                q.append(SubtreeTask(root_node=end_node, prefix_bits=new_prefix))

    return tree, results, n_features, max_depth


def _access_tutorial(tables: List[TableResult]) -> None:
    """
    Small printed tutorial on how to access:
      - number of tables (= number of subtrees)
      - per-table fields (root_node, prefix_bits, table, frontier_end)
      - unique structs: features, thresholds, outcome, index_map
    """
    print("\n[TUTORIAL] Accessing tables / subtrees")
    print("Number of tables (== subtrees):", len(tables))
    if not tables:
        return
    t0 = tables[0]
    print("First table example:")
    print("  root_node:", t0.root_node)
    print("  prefix_bits:", t0.prefix_bits if t0.prefix_bits else "<ROOT>")
    print("  entries (2^V):", len(t0.table))
    print("  sample path (node_ids):", t0.table[0])
    print("  sample frontier_end:", t0.frontier_end[0])
    print("Advanced access:")
    print("  for each table: for t in tables: ...")
    print("  for each entry: for i, nodes in enumerate(t.table): ...")
    print("  frontier_end[i] matches table[i].")
    print("\n[TUTORIAL] Accessing features/thresholds/outcome/index_map")
    print("  1) build table_id_by_root (see main), then:")
    print("     us = build_unique_structs_for_table(tree, t.table, t.frontier_end, V, table_id_by_root)")
    print("  2) available fields:")
    print("     us.unique_features[i]   -> list of V features (uint16)")
    print("     us.unique_thresholds[i] -> list of V thresholds (int16)")
    print("     us.unique_outcome[i]    -> int16 outcome of unique path i")
    print("     us.index_map[j]         -> maps raw entry j to unique path i")
    print("  3) access example:")
    print("     i = us.index_map[0]")
    print("     feats = us.unique_features[i]")
    print("     thrs  = us.unique_thresholds[i]")
    print("     out   = us.unique_outcome[i]")


# ----------------------------
# Unique structures per table
# ----------------------------

@dataclass
class UniqueStructs:
    unique_paths_nodes: List[List[int]]      # each path: V node_ids (NOT V+1)
    unique_features: List[List[int]]         # uint16
    unique_thresholds: List[List[int]]       # int16
    unique_outcome: List[int]                # int16, one per unique path
    index_map: List[int]                     # uint8, len = 2^V, maps i -> unique_index
    total_entries: int
    unique_count: int



def build_unique_structs_for_table(tree, table_paths: List[List[int]], frontier_end: List[int],
                                  V: int, table_id_by_root: Dict[int, int]) -> UniqueStructs:
    """
    - Uniqueness: use all V chunk nodes + the final frontier_end destination.
    - features/thresholds: store first V nodes (uint16 / int16).
    - outcome depends on frontier_end:
        leaf     -> 0 if pred==0 else -pred
        internal -> +table_id of the subtree (root=frontier_end)
    - index_map: uint8[2^V], maps raw index i -> unique_index
    """
    feats_arr = tree.feature
    thr_arr = tree.threshold
    values = tree.value

    cl = tree.children_left
    cr = tree.children_right

    def is_leaf(n: int) -> bool:
        return cl[n] == -1 and cr[n] == -1

    n_entries = 1 << V
    index_map: List[int] = [0] * n_entries

    # key -> unique_index
    # key = (tuple(V nodes), end_node)
    seen: Dict[Tuple[Tuple[int, ...], int], int] = {}

    unique_paths_nodes: List[List[int]] = []     # V nodes (debug/trace only)
    unique_features: List[List[int]] = []        # uint16, len = V
    unique_thresholds: List[List[int]] = []      # int16,  len = V
    unique_outcome: List[int] = []               # int16,  len = unique_count

    for i, nodes_full in enumerate(table_paths):
        # nodes_full has length V+1 (root included), with replicated leaf if needed.
        nodes_V = nodes_full[:V]          # all V nodes in the chunk (8 when V=8)

        end_node = int(frontier_end[i])  # node reached after V decisions (for outcome/subtree)

        key = (tuple(nodes_V), end_node)

        if key in seen:
            index_map[i] = seen[key]
            continue

        uidx = len(unique_paths_nodes)
        seen[key] = uidx
        index_map[i] = uidx

        # Store the V nodes only for traceability (optional but useful).
        unique_paths_nodes.append(list(nodes_V))

        # Features/thresholds: first V nodes.
        feature_list: List[int] = []
        threshold_list: List[int] = []

        for nid in nodes_V:
            f_raw = int(feats_arr[nid])
            t_raw = float(thr_arr[nid])

            if f_raw == -2:          # leaf
                f_u16 = 0
                t_i16 = INT16_MAX    # 32767
            else:
                f_u16 = to_uint16(f_raw)
                t_i16 = to_int16(t_raw)

            feature_list.append(f_u16)
            threshold_list.append(t_i16)

        unique_features.append(feature_list)       # len = V
        unique_thresholds.append(threshold_list)   # len = V

        # Outcome based on end_node.
        if is_leaf(end_node):
            counts = values[end_node][0]
            pred = int(counts.argmax())
            out = 0 if pred == 0 else -pred
        else:
            # A table must exist for this subtree.
            if end_node not in table_id_by_root:
                raise RuntimeError(
                    f"Subtree with root_node={end_node} does not have an associated table. "
                    f"(table_id_by_root is missing this key)"
                )
            out = int(table_id_by_root[end_node])  # positive

        # Clamp int16.
        if out < INT16_MIN:
            out = INT16_MIN
        elif out > INT16_MAX:
            out = INT16_MAX

        unique_outcome.append(out)

    return UniqueStructs(
        unique_paths_nodes=unique_paths_nodes,
        unique_features=unique_features,
        unique_thresholds=unique_thresholds,
        unique_outcome=unique_outcome,
        index_map=index_map,
        total_entries=n_entries,
        unique_count=len(unique_paths_nodes),
    )



# ----------------------------
# TXT dumps
# ----------------------------

def save_tables_txt(tables: List[TableResult], V: int, out_path: str) -> None:
    with open(out_path, "w") as f:
        f.write("# Decision Tree Path Tables (chunked)\n")
        f.write(f"# V = {V}\n")
        f.write("# Encoding per-chunk: MSB=local root, 1=true->left, 0=false->right\n")
        f.write("# Global path bits = prefix_bits + chunk_bits\n")
        f.write("# Each row: local_index | chunk_bits | global_bits | node_ids (len V+1)\n\n")

        tables_sorted = sorted(tables, key=lambda t: (len(t.prefix_bits), t.root_node))

        for k, t in enumerate(tables_sorted):
            f.write("=" * 80 + "\n")
            f.write(f"=== TABLE #{k} ===\n")
            f.write(f"subtree_root_node_id: {t.root_node}\n")
            f.write(f"prefix_bits (global): {t.prefix_bits if t.prefix_bits else '<ROOT>'}\n")
            f.write(f"entries: {1<<V}\n")
            f.write("-" * 80 + "\n")

            for i, nodes in enumerate(t.table):
                chunk_bits = format(i, f"0{V}b")
                global_bits = (t.prefix_bits + chunk_bits) if t.prefix_bits else chunk_bits
                nodes_str = ", ".join(map(str, nodes))
                f.write(f"{i:>6} | {chunk_bits} | {global_bits} | [{nodes_str}]\n")

            f.write("\n")


def save_unique_structs_txt(
    tree,
    tables: List[TableResult],
    V: int,
    out_path: str,
    table_id_by_root: Dict[int, int],
) -> None:
    with open(out_path, "w") as f:
        f.write("# Unique path structures per table\n")
        f.write(f"# V = {V}\n")
        f.write("# IMPORTANT: For each unique path we store ONLY V nodes (root included), NOT V+1.\n")
        f.write("# Per path: features[V], thresholds[V]\n")
        f.write("# Leaf mapping: feature=0 (uint16), threshold=32767 (int16)\n")
        f.write("# Thresholds are saved as int16 (rounded+clamped). Features as uint16.\n\n")


        tables_sorted = sorted(tables, key=lambda t: (len(t.prefix_bits), t.root_node))

        for k, t in enumerate(tables_sorted):
            us = build_unique_structs_for_table(tree, t.table, t.frontier_end, V, table_id_by_root)

            f.write("=" * 80 + "\n")
            f.write(f"=== UNIQUE STRUCTS FOR TABLE #{k} ===\n")
            f.write(f"subtree_root_node_id: {t.root_node}\n")
            f.write(f"prefix_bits (global): {t.prefix_bits if t.prefix_bits else '<ROOT>'}\n")
            f.write(f"table_entries: {us.total_entries}\n")
            f.write(f"unique_paths:  {us.unique_count}\n")
            f.write("-" * 80 + "\n")

            f.write("index_map (len=2^V):\n")
            for start in range(0, len(us.index_map), 32):
                chunk = us.index_map[start:start+32]
                f.write(f"{start:>4}: " + " ".join(f"{x:3d}" for x in chunk) + "\n")
            f.write("\n")

            # Print outcomes.
            f.write("unique_outcome (int16, len=unique_paths):\n")
            for start in range(0, len(us.unique_outcome), 32):
                chunk = us.unique_outcome[start:start+32]
                f.write(f"{start:>4}: " + " ".join(f"{x:6d}" for x in chunk) + "\n")
            f.write("\n")

            for i in range(us.unique_count):
                nodes = us.unique_paths_nodes[i]
                feats = us.unique_features[i]
                thrs = us.unique_thresholds[i]
                outc = us.unique_outcome[i]

                f.write(f"[path #{i}] outcome: {outc}\n")
                f.write(f"          nodes:      [{', '.join(map(str, nodes))}]\n")
                f.write(f"          features:   [{', '.join(map(str, feats))}]\n")
                f.write(f"          thresholds: [{', '.join(map(str, thrs))}]\n\n")


# ----------------------------
# CLI
# ----------------------------

def _format_bytes(n: int) -> str:
    # Simple formatter for readability.
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024.0:.2f} KiB"
    return f"{n / (1024.0 * 1024.0):.2f} MiB"

def compute_table_weight_bytes(V: int, unique_count: int) -> Dict[str, int]:
    # features/thresholds/outcome are int16 => 2 bytes each.
    # index_map is uint8 => 1 byte each.
    per_path_features = unique_count * (V) * 2
    per_path_thresholds = unique_count * (V) * 2
    per_path_outcome = unique_count * 2
    index_map_bytes = (1 << V) * 1
    total = per_path_features + per_path_thresholds + per_path_outcome + index_map_bytes
    return {
        "features": per_path_features,
        "thresholds": per_path_thresholds,
        "outcome": per_path_outcome,
        "index_map": index_map_bytes,
        "total": total,
    }

def generate_kernel():
    return f"""\
static const uint16_t lane_masks_u16[8][8] = {{
  {{1,0,0,0,0,0,0,0}},
  {{0,1,0,0,0,0,0,0}},
  {{0,0,1,0,0,0,0,0}},
  {{0,0,0,1,0,0,0,0}},
  {{0,0,0,0,1,0,0,0}},
  {{0,0,0,0,0,1,0,0}},
  {{0,0,0,0,0,0,1,0}},
  {{0,0,0,0,0,0,0,1}},
}};

static const uint16x8_t weights = {{128, 64, 32, 16, 8, 4, 2, 1}};

static inline uint16x8_t v_ones_u16(void)   {{ return vdupq_n_u16(1); }}
static inline uint16x8_t v_zeros_u16(void) {{ return vdupq_n_u16(0); }}

static inline uint16x8_t toggle_lane(uint16x8_t v, int lane){{
    uint16x8_t m = vld1q_u16(lane_masks_u16[lane]); // load 16B
    return veorq_u16(v, m);
}}

int kernel(int16_t *sample, const int16_t (*thresholds)[8], const uint16_t (*features)[8], const int16_t *outcome, const uint8_t *map, int index){{
    uint32_t zero;
    int16_t output;

    int starting_point = map[path_index[index]];
    uint16x8_t path = path_bits[index];

    while (1) {{
        int16x8_t  thresh = vld1q_s16(thresholds[starting_point]);
        uint16x8_t feat   = vld1q_u16(features[starting_point]);

        int16x8_t samp = vldrhq_gather_shifted_offset_s16(sample, feat);

        mve_pred16_t pred = vcmpleq_s16(samp, thresh);
        uint16x8_t pred_bit = vpselq_u16(v_ones_u16(), v_zeros_u16(), pred);

        mve_pred16_t cmp = vcmpneq_u16(pred_bit, path);

        if ((uint16_t)cmp == 0) {{
            output = outcome[starting_point];
            return (int)output;
        }}

        zero = __builtin_ctz((uint32_t)(uint16_t)cmp) >> 1;

        path = toggle_lane(path, (int)zero);
        path_bits[index] = path;

        path_index[index] = vmladavaq_u16(0, path, weights);
        starting_point = map[path_index[index]];
    }}
}}
"""

def testset_gen(csv_path, fh, n_rows):
    fh.write("RAM_BIG int16_t testset[TESTSIZE][SAMPLESIZE] = {\n")

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            if i >= n_rows:
                break
            # drop the last column (Label)
            int_row = [int(float(x)) for x in row[:-1]]
            rows.append(int_row)

    for r, row in enumerate(rows):
        values = ", ".join(str(v) for v in row)
        comma = "," if r < len(rows) - 1 else ""
        fh.write(f"   {{{values}}}{comma}\n")

    fh.write("};\n\n")

def generate_header(fh, full_struct, sample_size, test_size, csv_path=None, max_depth=None):
    fh.write("#ifndef INC_HEADER_H_\n")
    fh.write("#define INC_HEADER_H_\n\n")

    fh.write('#include "arm_mve.h"\n')
    fh.write('#include "string.h"\n')
    fh.write('#include "stdio.h"\n')
    fh.write('#include "stdlib.h"\n')
    fh.write('#include "stdbool.h"\n\n')

    fh.write(f"#define SUBTREES   {len(full_struct)}\n")
    fh.write(f"#define SAMPLESIZE {sample_size}\n")
    fh.write(f"#define TESTSIZE   {test_size}\n")
    if max_depth is not None:
        fh.write(f"#define MAX_DEPTH   {max_depth}\n")
    fh.write("#define DELTA      32767\n\n")

    fh.write('#define RAM_BIG __attribute__((section(".big_data"), aligned(32)))\n\n')

    fh.write("int16_t out;\n")
    fh.write(f"uint16x8_t path_bits[{len(full_struct)}];\n")
    fh.write(f"uint32_t path_index[{len(full_struct)}] = {{ {', '.join(['255'] * len(full_struct))} }};\n\n")

    for i in range(len(full_struct)):
        fh.write(f"const int16_t thresholds{i}[{len(full_struct[i].unique_thresholds)}][8] = {{\n")
        for j, thr in enumerate(full_struct[i].unique_thresholds):
            thr_str = ", ".join(str(t) for t in thr)
            comma = "," if j < len(full_struct[i].unique_thresholds) - 1 else ""
            fh.write(f"    {{{thr_str}}}{comma}\n")
        fh.write("};\n\n")
    
    for i in range(len(full_struct)):
        fh.write(f"const uint16_t features{i}[{len(full_struct[i].unique_features)}][8] = {{\n")
        for j, feat in enumerate(full_struct[i].unique_features):
            feat_str = ", ".join(str(f) for f in feat)
            comma = "," if j < len(full_struct[i].unique_features) - 1 else ""
            fh.write(f"    {{{feat_str}}}{comma}\n")
        fh.write("};\n\n")

    for i in range(len(full_struct)):
        outcomes = ", ".join(str(out) for out in full_struct[i].unique_outcome)
        fh.write(
            f"const int16_t outcome{i}[{len(full_struct[i].unique_outcome)}] = {{ {outcomes} }};\n"
        )
    fh.write("\n")

    for i in range(len(full_struct)):
        index_map = ", ".join(str(idx) for idx in full_struct[i].index_map)
        fh.write(
            f"const uint8_t index_map{i}[{len(full_struct[i].index_map)}] = {{ {index_map} }};\n"
        )
    fh.write("\n")

    n = len(full_struct)

    fh.write(
        "const int16_t  (*t_address[])[8] = { " +
        ", ".join(f"thresholds{i}" for i in range(n)) +
        " };\n\n"
    )

    fh.write(
        "const uint16_t (*f_address[])[8] = { " +
        ", ".join(f"features{i}" for i in range(n)) +
        " };\n\n"
    )

    fh.write(
        "const int16_t* out_address[] = { " +
        ", ".join(f"outcome{i}" for i in range(n)) +
        " };\n\n"
    )

    fh.write(
        "const uint8_t* map_address[] = { " +
        ", ".join(f"index_map{i}" for i in range(n)) +
        " };\n\n"
    )

    kernel = generate_kernel()
    fh.write(kernel)

    fh.write("static inline void init(void){\n")
    fh.write("   uint16x8_t v = vdupq_n_u16(1);\n")
    fh.write(f"   for (int i = 0; i < {len(full_struct)}; i++) {{\n")
    fh.write("        path_bits[i] = v;\n")
    fh.write("   }\n")
    fh.write("}\n\n")

    fh.write("static inline void inference(int16_t* sample){\n")
    fh.write(f"   out = kernel(sample, t_address[0], f_address[0], out_address[0], map_address[0], 0);\n")
    if max_depth > 8:
        fh.write("   while(out > 0){\n")
        fh.write("      out = kernel(sample, t_address[out], f_address[out], out_address[out], map_address[out], out);\n")
        fh.write("   }\n")
    fh.write("}\n\n")

    if csv_path is not None:
        testset_gen(csv_path, fh, test_size)

    fh.write("\n#endif /* INC_HEADER_H_ */\n")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate chunked 2^V tables for a DecisionTree and, for each table, "
            "extract DISTINCT paths (considering only the first V nodes) "
            "by building feature-index and threshold lists."
        )
    )
    ap.add_argument("joblib_path", help="Path to .joblib (bundle with 'model' key or direct model)")
    ap.add_argument("-V", "--depth", type=int, required=True, help="Chunk depth V")
    ap.add_argument("--tables_txt", default=None, help="TXT output for tables (default: *_chunked_V{V}.txt)")
    ap.add_argument("--structs_txt", default=None, help="TXT output for unique structures (default: *_structs_V{V}.txt)")
    ap.add_argument("--tutorial", action="store_true", help="Print a short table-access tutorial")
    ap.add_argument("--testpath", required=True, type=Path, help="Path to test CSV (last column = label).")
    args = ap.parse_args()

    joblib_path = args.joblib_path
    V = args.depth

    tree, tables, n_features, max_depth = generate_all_tables(joblib_path, V)

    tables_sorted = sorted(tables, key=lambda t: (len(t.prefix_bits), t.root_node))
    table_id_by_root = {t.root_node: idx for idx, t in enumerate(tables_sorted)}

    cl = tree.children_left
    cr = tree.children_right

    def is_leaf(n: int) -> bool:
        return cl[n] == -1 and cr[n] == -1

    all_leaf_nodes = {n for n in range(tree.node_count) if is_leaf(n)}

    leaves_reached_on_frontier = set()
    for t in tables:
        for n in t.frontier_end:
            if is_leaf(n):
                leaves_reached_on_frontier.add(n)

    print("tree.n_leaves =", tree.n_leaves)
    print("leaf nodes (scan) =", len(all_leaf_nodes))
    print("leaf nodes reached in at least one table frontier =", len(leaves_reached_on_frontier))

    missing = all_leaf_nodes - leaves_reached_on_frontier
    print("missing leaf nodes =", len(missing))
    if missing:
        print("example missing leaf ids:", list(sorted(missing))[:20])

    base, _ = os.path.splitext(joblib_path)

    tables_out = args.tables_txt or f"{base}_chunked_V{V}.txt"
    structs_out = args.structs_txt or f"{base}_structs_V{V}.txt"

    save_tables_txt(tables, V, tables_out)
    save_unique_structs_txt(tree, tables, V, structs_out, table_id_by_root)
    if args.tutorial:
        _access_tutorial(tables_sorted)

    # Useful summary prints.
    tables_sorted = sorted(tables, key=lambda t: (len(t.prefix_bits), t.root_node))
    print(f"[OK] Saved tables TXT:    {tables_out}")
    print(f"[OK] Saved structs TXT:   {structs_out}")
    print(f"[INFO] Number of tables: {len(tables_sorted)}")
    total_bytes_all = 0
    full_struct = []
    for k, t in enumerate(tables_sorted):
        us = build_unique_structs_for_table(tree, t.table, t.frontier_end, V, table_id_by_root)
        w = compute_table_weight_bytes(V, us.unique_count)
        total_bytes_all += w["total"]

        print(
            f"  - TABLE #{k}: root_node={t.root_node}, "
            f"unique_paths={us.unique_count}/{us.total_entries} "
            f"(reduction {100.0*(1.0 - us.unique_count/us.total_entries):.2f}%)"
        )
        print(
            f"    size: total={w['total']} B ({_format_bytes(w['total'])}) "
            f"[features={w['features']} B, thresholds={w['thresholds']} B, "
            f"outcome={w['outcome']} B, index_map={w['index_map']} B]"
        )
        full_struct.append(us)
    print(f"[INFO] Total size (all tables): {total_bytes_all} B ({_format_bytes(total_bytes_all)})")

    header_path = "weight.h"

    with open(header_path, "w") as f:
        generate_header(f, full_struct, sample_size=n_features, test_size=200, csv_path=args.testpath, max_depth=max_depth)

if __name__ == "__main__":
    main()
