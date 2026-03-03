from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Set, FrozenSet
from tqdm import tqdm

import networkx as nx

###############################################################################
# helpers
###############################################################################

def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def parse_input_tokens(s: str) -> Tuple[int,...]:
    return tuple(int(tok.split("_")[-1]) for tok in s.strip("<>").split("><"))


def powerset(iterable):
    """Return all possible subsets of the iterable except empty set and full set."""
    s = list(iterable)
    return [frozenset(subset) for r in range(1, len(s)) for subset in combinations(s, r)]


def extract_subsequence(full_sequence: Tuple[int], indices: FrozenSet[int]):
    """Extract a subsequence from the full sequence using the provided indices."""
    return tuple(full_sequence[i] for i in indices)


###############################################################################
# Substitution graph & coverage
###############################################################################

def get_behavior_map_per_indices(
    train_map: Dict[Tuple[int,...], int], 
    behavior_maps,  # thrice-nested mapping; will be updated in this function
    indices,
    full_length: int = 3,
): 
    if indices not in behavior_maps:
        # Initialize the behavior map
        behavior = defaultdict(dict)
        
        for full_seq, target in train_map.items():
            subseq = tuple(full_seq[i] if i in indices else -1 for i in range(full_length))  # masked full_seq: for convenience in `build_subst_graph`
            complement = tuple(full_seq[i] for i in range(full_length) if i not in indices)
            behavior[subseq][complement] = target
        
        behavior_maps[indices] = dict(behavior)
    
    return behavior_maps[indices]


def build_subst_graph(
    behavior_maps: Dict[Tuple[int,...], Dict],  # thrice-nested mapping; assumed to be generated in `get_behavior_map_per_indices`.
    all_tuples: List[Tuple[int, ...]],
    min_evidence: int = 1,
    full_length: int = 3,
) -> nx.Graph:
    """Build an indices-specific substitution graph."""
    
    G = nx.Graph()
    for tr in all_tuples:
        G.add_node(tr)
        
    for indices, behavior in behavior_maps.items():
        indices_complement = sorted(set(range(full_length)) - indices)
        all_complements = set([tuple(tup[i] for i in indices_complement) for tup in all_tuples])

        for (subseq1, compl_map1), (subseq2, compl_map2) in combinations(behavior.items(), 2):
            shared_complements = set(compl_map1) & set(compl_map2)  # co-occurrences
            if len(shared_complements) >= min_evidence:
                functionally_k_equivalent = True
                for comp in shared_complements:
                    if compl_map1[comp] != compl_map2[comp]:
                        functionally_k_equivalent = False
                        break
                if functionally_k_equivalent:  # all co-occurrences (# >= k) are consistent; functionally k-equivalent
                    seq1, seq2 = list(subseq1), list(subseq2)
                    for comp in all_complements:
                        for j, idx in enumerate(indices_complement):
                            seq1[idx] = seq2[idx] = comp[j]
                        G.add_edge(tuple(seq1), tuple(seq2))
    
    logging.info(f"Substitution graph (indices=): |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    return G


def compute_coverage(G: nx.Graph, train_tuples: List[Tuple[int,...]]) -> Set[Tuple[int,...]]:
    covered = set()
    for tr in train_tuples:
        if tr not in covered:
            conn = nx.node_connected_component(G, tr)
            covered.update(conn)
    logging.info(f"Covered nodes: {len(covered)}")
    return covered


###############################################################################
# main
###############################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--visualise", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--min_evidence", type=int, default=1,
                   help="minimum evidence for equivalence classes")
    ap.add_argument("--k_sweep", action="store_true",
                   help="Run multiple min_evidence values (1..max_k) and report coverage")
    ap.add_argument("--max_k", type=int, default=8,
                   help="maximum min_evidence value; used only when `--k_sweep` is activated")
    ap.add_argument("--target_indices", type=int, nargs="*", default=None,
                   help="Compute coverage for specific index set (e.g.,`--target_indices 0 1`)")
    args = ap.parse_args()

    setup_logging(args.debug)

    def jload(name):
        with open(os.path.join(args.data_dir, name), "r", encoding="utf-8") as f:
            return json.load(f)

    train: List[Dict[str, str]] = jload("train.json")
    test: List[Dict[str, str]] = jload("test.json")

    parse = parse_input_tokens
    
    ## Remark - types of test examples (n-hop task):
    ## `type_0`: all n atmoic facts are observed independently  - candidate elements inside the coverage boundary (≈ "In-Domain Closure")
    ## `type_m` (m>=1): For binary representation (b1, ..., bn) of (m-1)>=0, the i-th atomic fact is observed iff bi == 1  - necessarily OOD
    train_tuples = [parse(it["input_text"]) for it in train]
    test_tuples = [parse(it["input_text"]) for it in test if it["type"] == "type_0"]
    # test_tuples = [d for d in test_tuples if d not in train_tuples]  # Remove train examples if needed
    logging.info(f"train tuples: {len(train_tuples)}  test tuples: {len(test_tuples)}")
    assert len(train_tuples) == len(set(train_tuples)), "duplicate tuples in train.json"
    assert len(test_tuples) == len(set(test_tuples)), "duplicate tuples in test.json"

    tuple2t = {parse(it["input_text"]): parse(it["target_text"].replace("</a>", ""))[-1] for it in train + test}  # train_map + test_map
    train_map = {tr: tuple2t[tr] for tr in train_tuples}

    # get `full_len`: (n+1) for n-hop task.
    for example in tuple2t:
        full_len = len(example)
        break

    type0_indices = [i for i, item in enumerate(test) if item.get("type") == "type_0"]  # effectively the same as range(len(test)).
    type0_total = len(type0_indices)

    # Cache the behavior maps for efficiency
    logging.info("Caching behavior maps for all subsets...")
    behavior_maps = {}
    if args.target_indices is None:
        all_subsets = powerset(range(full_len))
        for indices in all_subsets:
            get_behavior_map_per_indices(train_map, behavior_maps, indices, full_len)
    else:  # args.target_indices: List[int,...]
        get_behavior_map_per_indices(train_map, behavior_maps, frozenset(args.target_indices), full_len)
        target_indices_strs = ",".join(map(str, args.target_indices))

    if args.k_sweep:
        k_sweep_results = {}
        
        # For tracking coverage of each example at each k
        # Map from test example index to highest k where it's covered
        coverage_by_example = {i: 0 for i in type0_indices}  # Default to uncovered
        
        # Run the k-sweep
        k_values = range(1, args.max_k+1)
        
        for k in tqdm(k_values, desc="k-sweep progress"):
            logging.info(f"Running coverage analysis with min_evidence = {k}")
            
            # Build graph and compute coverage
            G = build_subst_graph(behavior_maps, train_tuples + test_tuples, k, full_length=full_len)
            covered = compute_coverage(G, train_tuples)

            # Count coverage for type_0 and track individual test examples
            type0_covered = 0
            
            for i in type0_indices:
                if parse(test[i]["input_text"]) in covered:
                    type0_covered += 1
                    # Update this example's coverage threshold to current k
                    coverage_by_example[i] = k
            
            # Calculate percentage
            coverage_pct = (type0_covered / type0_total * 100) if type0_total > 0 else 0
            k_sweep_results[k] = (type0_covered, coverage_pct)
            
            logging.info(f"k={k}: {type0_covered}/{type0_total} type_0 covered ({coverage_pct:.2f}%)")
        
        # Write results to JSON file
        os.makedirs("k_sweep_results", exist_ok=True)
        k_sweep_file = os.path.join("k_sweep_results", f"{args.data_dir.split('/')[-2]}{'_full' if args.target_indices is None else '_I='+target_indices_strs}.json")
        with open(k_sweep_file, "w", encoding="utf-8") as f:
            json.dump(k_sweep_results, f, indent=2)
        
        logging.info(f"k-sweep results written to {k_sweep_file}")
        
        # Create modified test data with coverage thresholds
        new_test_data = test.copy()
        coverage_threshold_counts = {i: 0 for i in range(args.max_k+1)}  # Count examples for each threshold
        
        # Create entries with coverage threshold types
        for i in type0_indices:
            threshold = coverage_by_example[i]
            coverage_threshold_counts[threshold] += 1
            
            # Create a duplicate entry with the coverage threshold type
            duplicate_entry = test[i].copy()
            duplicate_entry["type"] = f"covered_{threshold}"
            new_test_data.append(duplicate_entry)
        
        # Write the modified test file
        threshold_test_file = os.path.join(args.data_dir, f"test_annotated{'_full' if args.target_indices is None else '_I='+target_indices_strs}.json")
        with open(threshold_test_file, "w", encoding="utf-8") as f:
            json.dump(new_test_data, f, indent=2)
        
        # Log threshold distribution
        logging.info("Coverage threshold distribution:")
        for threshold, count in coverage_threshold_counts.items():
            logging.info(f"  covered_{threshold}: {count} examples")
        
        logging.info(f"Enhanced test data with coverage thresholds written to {threshold_test_file}")


    else:
        # Build full substitution graph
        G = build_subst_graph(behavior_maps, train_tuples + test_tuples, args.min_evidence, full_length=full_len)

        # Compute coverage
        covered = compute_coverage(G, train_tuples)

        # Annotate test.json
        for it in test:
            it["coverage"] = bool(parse(it["input_text"]) in covered)
        
        # Coverage report by type
        totals = defaultdict(int)
        hits = defaultdict(int)

        for it in test:
            typ = it.get("type", "UNK")
            totals[typ] += 1
            if it["coverage"]:
                hits[typ] += 1

        for typ in sorted(totals):
            pct = 100.0 * hits[typ] / totals[typ]
            logging.info(f"Coverage [{typ}] : {hits[typ]} / {totals[typ]}  ({pct:.2f}%)")

        # Write annotated test file with appropriate suffix
        annotated_file = os.path.join(args.data_dir, f"test_annotated{'_full' if args.target_indices is None else '_I='+target_indices_strs}.json")
        with open(annotated_file, "w", encoding="utf-8") as f:
            json.dump(test, f, indent=2)
        logging.info(f"{annotated_file} written")

        # Visualization logic
        if args.visualise:
            # ------------------------------------------------------------------
            # 1.  keep only the tuples we want to see
            # ------------------------------------------------------------------
            type0_tuples = []
            logging.info("Selecting type_0 tuples for visualization...")
            for it in test:
                if it.get("type") == "type_0":
                    tr = parse(it["input_text"])
                    assert tr not in train_tuples, f"duplicate found: {tr!r}"
                    type0_tuples.append(tr)

            train_set = set(train_tuples)
            type0_set = set(type0_tuples)
            
            # ------------------------------------------------------------------
            # 2.  build a *visualisation* graph on this subset (with caching)
            # ------------------------------------------------------------------
            viz_tuples = train_tuples + type0_tuples
            
            # Create cache directory if it doesn't exist
            cache_dir = os.path.join(args.data_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Cache file path for the graph
            cache_file = os.path.join(cache_dir, f"viz_graph_min{args.min_evidence}.pkl")
            
            # Try to load from cache first
            if os.path.exists(cache_file):
                logging.info(f"Loading visualization graph from cache: {cache_file}")
                import pickle
                with open(cache_file, 'rb') as f:
                    G_viz = pickle.load(f)
                logging.info(f"Loaded graph with |V|={G_viz.number_of_nodes()}, |E|={G_viz.number_of_edges()}")
            else:
                logging.info("Building visualization graph (this may take a while)...")
                G_viz = build_subst_graph(behavior_maps, viz_tuples, args.min_evidence, full_length=full_len)
                
                # Save to cache for future use
                logging.info(f"Saving visualization graph to cache: {cache_file}")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(G_viz, f)

            # ------------------------------------------------------------------
            # 3. compute layout with optimized algorithm
            # ------------------------------------------------------------------
            # Try to use ForceAtlas2 for faster layout if available
            try:
                from fa2_modified import ForceAtlas2
                logging.info("Using ForceAtlas2 for graph layout...")
                
                # Cache the layout calculation
                layout_cache = os.path.join(cache_dir, f"layout_min{args.min_evidence}.pkl")
                
                if os.path.exists(layout_cache):
                    logging.info(f"Loading layout from cache: {layout_cache}")
                    with open(layout_cache, 'rb') as f:
                        pos = pickle.load(f)
                else:
                    logging.info(f"Computing optimized layout for {G_viz.number_of_nodes()} nodes...")
                    
                    # Use ForceAtlas2 for larger graphs
                    if G_viz.number_of_nodes() > 50:
                        forceatlas2 = ForceAtlas2(
                            outboundAttractionDistribution=True,
                            linLogMode=False,
                            adjustSizes=False,
                            edgeWeightInfluence=1.0,
                            jitterTolerance=1.0,
                            barnesHutOptimize=True,
                            barnesHutTheta=1.2,
                            multiThreaded=False,  # Set to True if your installation supports it
                            scalingRatio=2.0,
                            strongGravityMode=False,
                            gravity=1.0,
                            verbose=False
                        )
                        
                        # Compute layout with progress indication
                        logging.info("Computing ForceAtlas2 layout...")
                        pos = {}
                        iterations = 500
                        
                        # Compute initial positions using spring_layout with few iterations
                        initial_pos = nx.spring_layout(G_viz, seed=0, iterations=5)
                        
                        # Run ForceAtlas2 with progress bar
                        for i in tqdm(range(iterations), desc="Layout iterations"):
                            pos = forceatlas2.forceatlas2_networkx_layout(
                                G_viz, pos=initial_pos if i == 0 else pos, iterations=1
                            )
                    else:
                        # For smaller graphs, spring_layout works well
                        logging.info("Computing spring layout...")
                        pos = nx.spring_layout(G_viz, seed=0, iterations=100)
                        
                    # Save layout to cache
                    logging.info(f"Saving layout to cache: {layout_cache}")
                    with open(layout_cache, 'wb') as f:
                        pickle.dump(pos, f)
            except ImportError:
                logging.info("ForceAtlas2 not available, using spring_layout...")
                pos = nx.spring_layout(G_viz, seed=0)

            # ------------------------------------------------------------------
            # 4.  prepare visualization with batched processing
            # ------------------------------------------------------------------
            import plotly.graph_objects as go
            
            if not os.path.exists('coverage_visualization'):
                os.makedirs('coverage_visualization', exist_ok=True)
            plot_save_dir = os.path.join('coverage_visualization', f"{args.data_dir.split('/')[-2]}_full_min{args.min_evidence}.html")
                
            logging.info(f"Preparing visualization data...")

            buckets = {                 # label → 3 empty lists (x,y,text)
                "train (covered)"       : ([], [], []),
                "type_0 ✓ covered"      : ([], [], []),
                "type_0 ✗ uncovered"    : ([], [], []),
            }

            # Process nodes in batches to avoid memory issues with large graphs
            BATCH_SIZE = 5000
            node_batches = [list(G_viz.nodes())[i:i+BATCH_SIZE]
                        for i in range(0, len(G_viz.nodes()), BATCH_SIZE)]
            
            for batch in node_batches:
                for n in batch:
                    if n not in pos:
                        logging.warning(f"Node {n} missing from layout positions, skipping")
                        continue
                        
                    x, y = pos[n]
                    txt = str((*n, tuple2t[n]))

                    if n in train_set:
                        label = "train (covered)"
                    elif n in type0_set:
                        if n in covered:
                            label = "type_0 ✓ covered"
                        else:
                            label = "type_0 ✗ uncovered"
                    else:
                        continue  # Skip other node types

                    buckets[label][0].append(x)
                    buckets[label][1].append(y)
                    buckets[label][2].append(txt)

            STYLE = {
                "train (covered)"    : dict(symbol="diamond", size=8, color="#1f77b4"),
                "type_0 ✓ covered"   : dict(symbol="circle",  size=7, color="#2ca02c"),
                "type_0 ✗ uncovered" : dict(symbol="x",       size=8, color="#d62728"),
            }
            
            # Create traces only for non-empty buckets
            node_traces = []
            for label, (xs, ys, texts) in buckets.items():
                if not xs:  # Skip empty buckets
                    continue
                    
                node_traces.append(
                    go.Scatter(
                        x=xs, y=ys,
                        mode="markers",
                        name=label,
                        marker=STYLE[label],
                        text=texts,
                        hovertemplate="%{text}",
                    )
                )
            
            # Process edges in batches to avoid memory issues
            logging.info("Processing edges for visualization...")
            edge_x, edge_y = [], []
            
            EDGE_BATCH_SIZE = 10000
            edge_list = list(G_viz.edges())
            edge_batches = [edge_list[i:i+EDGE_BATCH_SIZE] 
                        for i in range(0, len(edge_list), EDGE_BATCH_SIZE)]
            
            for batch in tqdm(edge_batches, desc="Processing edge batches"):
                batch_x, batch_y = [], []
                for u, v in batch:
                    if u not in pos or v not in pos:
                        continue
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    batch_x.extend([x0, x1, None])
                    batch_y.extend([y0, y1, None])
                edge_x.extend(batch_x)
                edge_y.extend(batch_y)
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(width=0.4, color="#bbbbbb"),
                hoverinfo="skip",
                showlegend=False,
            )

            # ------------------------------------------------------------------
            # 5.  create and save the figure
            # ------------------------------------------------------------------
            logging.info("Creating Plotly figure...")
            fig = go.Figure(
                data=[edge_trace] + node_traces,
                layout=go.Layout(
                    title=f"Full Substitution Graph • min_evidence={args.min_evidence} • {len(G_viz.nodes())} nodes, {len(G_viz.edges())} edges",
                    hovermode="closest",
                    margin=dict(l=20, r=20, t=60, b=20),
                    xaxis=dict(visible=False), 
                    yaxis=dict(visible=False),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255, 255, 255, 0.8)"
                    ),
                ),
            )
            
            logging.info(f"Writing HTML graph to {plot_save_dir}...")
            fig.write_html(plot_save_dir)
            logging.info(f"HTML visualization saved → {plot_save_dir}")


if __name__ == "__main__":
    main()
