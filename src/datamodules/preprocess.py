from collections import ChainMap
from pathlib import Path

import torch
import joblib
import pandas as pd
import networkx as nx
from torch_geometric.data import Data

from src.datamodules.graphs import extract_induced_subgraph
from src.datamodules.features import get_edge_features, get_node_features
from src.utils import paths


def read_pydot(
    path: str | Path, as_dict: bool = False
) -> nx.DiGraph | dict[str, nx.DiGraph]:
    """
    Returns a nx.DiGraph object representing a biological pathway.
    Removes an undesired '\\n' node if present (artifact from .dot reading in networkx).
        Parameters:
            path (str,pathlih.Path):
                the path where the .dot file is located.

        Returns:
            G (nx.DiGraph):
                a nx.DiGraph directed graph.
    """
    path = Path(path)
    G = nx.DiGraph(nx.nx_pydot.read_dot(path))
    if "\\n" in G.nodes:
        G.remove_node("\\n")

    if as_dict:
        return {path.stem: G}

    return G


def load_single_pathway(pathway_id: str, as_dict: bool = False):
    """Loads a pathway with a given id."""
    path = paths.PATHWATYS_DIR / f"{pathway_id}.dot"
    return read_pydot(path, as_dict=as_dict)


def load_pathways(as_dict: bool = True) -> list[nx.DiGraph] | ChainMap[str, nx.DiGraph]:
    """ "Loads a collection of patways in parallel."""
    ex = joblib.Parallel(verbose=1, n_jobs=-1)
    job = joblib.delayed(read_pydot)
    pathways = ex(job(p, as_dict=as_dict) for p in paths.PATHWAY_PATHS)

    if as_dict:
        return ChainMap(*pathways)  # type: ignore

    return pathways  # type: ignore


def load_subgraphs_df(path: str) -> pd.DataFrame:
    """Loads the subgraphs data and attaches the original graph they originate from."""
    pathways = load_pathways(as_dict=True)
    subgraphs_data = pd.read_csv(path, keep_default_na=False)
    subgraphs_data["Graph"] = subgraphs_data.PathwayID.map(lambda gid: pathways[gid])
    return subgraphs_data[["Graph", "PathwayID", "Input", "Output", "Property"]]


def featurize_graph(
    G: nx.DiGraph,
    input_species: str,
    output_species: str,
    prop: int | None,
) -> Data:
    """Transforms a graph into a Data object."""

    y = None
    if prop is not None:
        y = torch.FloatTensor([prop])

    x = get_node_features(G, input_species, output_species)
    edge_index, edge_attr = get_edge_features(G)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def preprocess_single_graph(
    G: nx.DiGraph,
    pathway_id: str,
    input_species: str,
    output_species: str,
    prop: int | None,
) -> Data | None:
    """Applies preprocessing to a subgraphs dataframe entry."""
    subG = extract_induced_subgraph(G, input_species, output_species)
    if subG is None:
        return None

    data = featurize_graph(subG, input_species, output_species, prop)
    meta = {
        "pathway_id": pathway_id,
        "input": input_species,
        "output": output_species,
        "property": prop,
    }
    return meta, data  # type: ignore


def preprocess_graphs(subgraphs_df: pd.DataFrame) -> list[tuple]:
    """Preprocesses all the subgraphs and transforms them into Data objects."""
    ex = joblib.Parallel(verbose=1, n_jobs=-1)
    job = joblib.delayed(preprocess_single_graph)
    data_list = ex(job(*row) for row in subgraphs_df.to_numpy())
    return [d for d in data_list if d]  # type: ignore


def retrieve_pathway_data(
    name: str,
    pathway_id: str,
    input_species: str,
    output_species: str,
):
    path = paths.DATA_DIR / name / "raw" / "data.csv"
    subgraphs_data = pd.read_csv(path, keep_default_na=False)
    subgraphs_data = subgraphs_data[subgraphs_data.PathwayID == pathway_id]
    subgraphs_data = subgraphs_data[subgraphs_data.Input == input_species]
    subgraphs_data = subgraphs_data[subgraphs_data.Output == output_species]

    assert subgraphs_data.shape[0] == 1

    row = subgraphs_data.iloc[0]
    return {
        "pathway_id": row.PathwayID,
        "input": row.Input,
        "output": row.Output,
        "property": row.Property,
    }


def get_knockout_data(
    name: str,
    pathway_id: str,
    input_species: str,
    output_species: str,
):
    G = load_single_pathway(pathway_id)
    subG = extract_induced_subgraph(G, input_species, output_species)  # type: ignore

    if subG is None:
        raise ValueError(
            f"No data for {pathway_id} ({input_species}, {output_species})"
        )

    meta = retrieve_pathway_data(name, pathway_id, input_species, output_species)
    meta["arc"] = None

    data = featurize_graph(subG, input_species, output_species, meta["property"])

    data_list, info = [data], [meta]

    for edge in sorted(subG.edges()):
        G_copy = nx.DiGraph(subG.copy())
        G_copy.remove_edges_from([edge])

        if nx.is_connected(G_copy.to_undirected()):
            meta = retrieve_pathway_data(
                name, pathway_id, input_species, output_species
            )
            meta["arc"] = edge
            data = featurize_graph(
                G_copy, input_species, output_species, meta["property"]
            )
            data_list.append(data)
            info.append(meta)

    return data_list, info, nx.DiGraph(subG)


def get_predict_data(
    name: str,
    pathway_id: str,
    input_species: str,
    output_species: str,
):
    G = load_single_pathway(pathway_id)
    subG = extract_induced_subgraph(G, input_species, output_species)  # type: ignore

    if subG is None:
        raise ValueError(
            f"No data for {pathway_id} ({input_species}, {output_species})"
        )
    meta = retrieve_pathway_data(name, pathway_id, input_species, output_species)
    data = featurize_graph(subG, input_species, output_species, meta["property"])
    return meta, data
