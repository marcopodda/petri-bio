import math

from typing import List
from pathlib import Path

import numpy as np
import networkx as nx


def print_colored_subgraph(
    G: nx.DiGraph,
    name: str,
    input_species: str,
    output_species: str,
    prop: int,
    prediction: np.ndarray,
    edges: List,
    output_path: Path,
):
    label = f"{name}: {prop:.2f}\n"
    label += f"Prediction: {prediction[0]:.2f}"
    G.graph["label"] = label  #  type: ignore

    intensities = [elem - prediction[0] for elem in prediction[1:]]

    reds, blues = {}, {}
    for edge, intensity in zip(edges[1:], intensities):  #  type: ignore
        if intensity > 0:
            reds[edge] = intensity
        else:
            blues[edge] = intensity

    for edge in G.edges():
        G.edges[edge]["label"] = ""

    for key, value in reds.items():
        G.edges[key]["colorscheme"] = "reds7"
        if math.ceil(value * 10) + 2 > 7:
            G.edges[key]["color"] = "7"
            G.edges[key]["label"] = f"{value:.2f}"
        else:
            G.edges[key]["color"] = math.ceil(value * 10) + 2
            G.edges[key]["label"] = f"{value:.2f}"

    for key, value in blues.items():
        G.edges[key]["colorscheme"] = "blues7"
        if math.ceil(abs(value) * 10) + 2 > 7:
            G.edges[key]["color"] = "7"
            G.edges[key]["label"] = f"{value:.2f}"
        else:
            G.edges[key]["color"] = math.ceil(abs(value) * 10) + 2
            G.edges[key]["label"] = f"{value:.2f}"

    G.nodes[input_species]["color"] = "green4"
    G.nodes[output_species]["color"] = "gold2"
    G.nodes[input_species]["label"] = input_species + "\nIN"
    G.nodes[output_species]["label"] = output_species + "\nOUT"

    drawing = nx.nx_agraph.to_agraph(G)
    drawing.draw(str(output_path), prog="dot")
