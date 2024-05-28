import numpy as np
import networkx as nx

from intervaldict import IntervalDict

from msr.msr import SegmentLayer


def segment_map_to_graph(segment_map):
    nlevels = len(segment_map)

    g = nx.DiGraph()

    # insert nodes
    for level in range(nlevels):
        segment_layer = segment_map[level]
        segment_start = segment_layer.data["start"]
        segment_size = segment_layer.data["size"]
        segment_ci_value = segment_layer.data["isign"]
        segment_sfc_score = segment_layer.data["sfc_score"]

        nsegments = len(segment_start)
        segment_end = segment_start + segment_size

        for iseg in range(nsegments):
            prune = (
                (segment_ci_value[iseg] == 1)
                #or (np.isclose(0.0, segment_sfc_score[iseg]))
                # NOTE isclose is slow
                or (segment_sfc_score[iseg] == 0)
            )

            # NOTE: does it matter in the end whether we insert these pruned
            # segments? Couldn't we kick out these segments already at the
            # beginning?
            # NOTE: SURPRISE MOTHERFUCKER, IT DOES MATTER; because pruning can
            # influence across many levels, but segments are connected only
            # from level i to (i + 1)
            nodedata = {
                "startbin": segment_start[iseg],
                "endbin": segment_end[iseg],
                "size": segment_size[iseg],
                "sign": segment_ci_value[iseg],
                "score": segment_sfc_score[iseg],
                "pruned": prune,
            }
            g.add_node((level, iseg), **nodedata)

    # create edges from child to parent
    for childlevel in range(nlevels - 1):
        child_layer = segment_map[childlevel]

        parentlevel = childlevel + 1

        child_segment_start = child_layer.data["start"]
        child_segment_end = (
            child_layer.data["start"]
            + child_layer.data["size"]
        )
        parent_segment_start = segment_map[parentlevel].data["start"]

        span_start = np.searchsorted(parent_segment_start, child_segment_start, side="right") - 1
        span_end = np.searchsorted(parent_segment_start, child_segment_end, side='right')
        for ichildseg, (sstart, send) in enumerate(zip(span_start, span_end)):
            for iparentseg in range(sstart, send):
                g.add_edge((childlevel, ichildseg), (parentlevel, iparentseg))

    # Reduce graph size by pruning loose branches (node is tagged "pruned" and
    # has no children), layer by layer
    # Note that this seems the only acceptable situation where nodes can be
    # pre-emptively pruned
    # TODO: not sure whether this doesn't lead to unexpected behavior on the
    # side of the caller ... (e.g. because len(segment_graph) can suddenly be
    # 0)
    for ilayer in range(len(segment_map)):
        for iseg in range(len(segment_map[ilayer])):
            node = (ilayer, iseg)
            if g.nodes[node]["pruned"] and not any(g.predecessors(node)):
                g.remove_node(node)

    return g


def prune_segment_graph(segment_graph, segment_counts, *, T):
    """
    Note that `segment_map` is only used to extract "metadata";
        - the number of layers
        - the number of segments per layer
    """
    nlayers = len(segment_counts)

    # prune upwards:
    for childlevel in range(nlayers - 1):
        nchildren = segment_counts[childlevel]
        for ichildseg in range(nchildren):
            childnode = (childlevel, ichildseg)
            if childnode not in segment_graph:
                continue

            childnodedata = segment_graph.nodes[childnode]
            if childnodedata["pruned"]:
                continue

            for _, (parentlevel, iparentseg) in nx.edge_bfs(segment_graph, (childlevel, ichildseg)):
                parentnode = (parentlevel, iparentseg)
                assert parentnode in segment_graph
                parentnodedata = segment_graph.nodes[parentnode]
                if parentnodedata["score"] <= (childnodedata["score"] / T):
                    parentnodedata["pruned"] = True

    # prune downwards:
    for parentlevel in reversed(range(1, nlayers)):
        nparents = segment_counts[parentlevel]
        for iparentseg in range(nparents):
            parentnode = (parentlevel, iparentseg)
            if parentnode not in segment_graph:
                continue

            parentnodedata = segment_graph.nodes[parentnode]
            if parentnodedata["pruned"]:
                continue

            for (childlevel, ichildseg), _, _ in nx.edge_bfs(segment_graph, (parentlevel, iparentseg), orientation="reverse"):
                childnodedata = segment_graph.nodes[(childlevel, ichildseg)]
                if childnodedata["score"] < parentnodedata["score"]:
                    childnodedata["pruned"] = True

    # prune downwards, choose highest-level segment by disregarding score (necessary due to slack in earlier step):
    for parentlevel in reversed(range(1, nlayers)):
        nparents = segment_counts[parentlevel]
        for iparentseg in range(nparents):
            parentnode = (parentlevel, iparentseg)
            if parentnode not in segment_graph:
                continue

            parentnodedata = segment_graph.nodes[parentnode]
            if parentnodedata["pruned"]:
                continue

            for (childlevel, ichildseg), _, _ in nx.edge_bfs(segment_graph, (parentlevel, iparentseg), orientation="reverse"):
                childnodedata = segment_graph.nodes[(childlevel, ichildseg)]
                childnodedata["pruned"] = True

    return segment_graph


class NoUnprunedNodes(Exception): pass


def flatten_segment_graph(g):
    unpruned_nodes = [
        k for (k, v) in
        g.nodes(data="pruned")
        if not v
    ]

    if not unpruned_nodes:
        # not really something to do, since interval datastructure doesn't work empty intervals
        raise NoUnprunedNodes

    datas = [g.nodes[node] for node in unpruned_nodes]
    starts, ends = map(np.array, (zip(*((d['startbin'], d['endbin']) for d in datas))))

    tree = IntervalDict(starts, ends)

    return tree, unpruned_nodes


def validate_single_node_per_position(tree):
    start, stop = tree.get_bounds()

    return all(
        len(stepval) <= 1
        for _, _, stepval in tree.get_steps(start, stop)
    )


def segment_graph_unpruned_subgraph(segment_graph):
    return segment_graph.subgraph([
        node for node in segment_graph.nodes
        if not segment_graph.nodes[node]["pruned"]
    ])


def flat_score(segment_map, sv, l=None, out=None, attr="sfc_score", sign=True):
    if not ((l is None) ^ (out is None)):
        raise ValueError()

    # elide, let caller run this
    # assert validate_single_node_per_position(sv)

    if out is None:
        out = np.zeros(l)
    else:
        assert out.ndim == 1
        l = len(out)

    for stepstart, stepend, stepnodes in sv[0:l]:
        if stepnodes:
            stepnode = next(iter(stepnodes))  # assuming only one stepnode, elide assertion
            level, isegment = stepnode

            if sign:
                stepval = (
                    segment_map[level].data[attr][isegment]
                    * (int(segment_map[level].data["isign"][isegment]) - 1)
                )
            else:
                stepval = segment_map[level].data[attr][isegment]

            out[stepstart:stepend] = stepval

    return out
