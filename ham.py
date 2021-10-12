import netket as nk
import numpy as np

sz_sz = np.asarray(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]
)
sxy_sxy = np.asarray(
    [
        [0, 0, 0, 0],
        [0, 0, 2, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
    ]
)


def Triangular(L, pbc):
    def k(i, j):
        return (i % L) * L + (j % L)

    def colored(i, j, c):
        if i < j:
            return i, j, c * 2
        else:
            return j, i, c * 2 + 1

    # Edges are undirected
    edges = []
    for i in range(L):
        for j in range(L - 1 + pbc):
            edges.append(colored(k(i, j), k(i, j + 1), 0))
    for i in range(L - 1 + pbc):
        for j in range(L):
            edges.append(colored(k(i, j), k(i + 1, j), 0))
    for i in range(L - 1 + pbc):
        for j in range(L - 1 + pbc):
            edges.append(colored(k(i + 1, j + 1), k(i, j), 1))

    graph = nk.graph.Graph(edges=edges, n_nodes=L**2)
    return graph


def HeisenbergTriangular(hilbert, graph, J, sign_rule):
    operators = []
    acting_on = []
    for i, j, c in graph.edges(return_color=True):
        assert i < j
        assert c in [0, 1, 2, 3]
        if sign_rule == "none":
            operators.append(J * (sz_sz + sxy_sxy))
            acting_on.append((i, j))
        elif sign_rule == "mars":
            if c // 2 == 0:
                operators.append(J * (sz_sz - sxy_sxy))
            else:
                operators.append(J * (sz_sz + sxy_sxy))
            acting_on.append((i, j))
        else:
            raise ValueError(f"Unknown sign_rule: {sign_rule}")

    H = nk.operator.LocalOperator(
        hilbert=hilbert, operators=operators, acting_on=acting_on
    )
    return H
