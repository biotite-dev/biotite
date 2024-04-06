import pandas as pd
import argparse
import json
import numpy as np

def process(input, output, chain):
    data = pd.read_csv(input)
    # Only retain rows with basepair annotation
    data = data[data['Leontis-Westhof'].notna()]

    output_list = []

    for _, row in data.iterrows():
        nucleotides = [row['Nucleotide 1'], row['Nucleotide 2']]

        # Extract the Leontis-Westhof annotation
        lw_string = row['Leontis-Westhof']

        # Some interactions are labelled with `n` for near. These are
        # ignored
        if lw_string[0] == 'n':
            continue

        # Get edge annotations from string
        edges = [lw_string[-2], lw_string[-1]]
        
        # Dont allow unspecified edges in test data
        if '.' in edges:
            continue

        res_ids = [None]*2
        for i, nucleotide in enumerate(nucleotides):
            nucleotide_list = nucleotide.split('.')

            # if the nucleotide is not part of the specified chain, skip
            # base pair
            if nucleotide_list[1] != chain:
                break

            res_ids[i] = nucleotide_list[3]

        if None in res_ids:
            continue

        for i, edge in enumerate(edges):
            if edge == 'W':
                edges[i] = 1
            if edge == 'H':
                edges[i] = 2
            if edge == 'S':
                edges[i] = 3

        # Lower residue id on the left, higher residue id on the right
        res_ids = np.array(res_ids, dtype=int)
        edges = np.array(edges, dtype=int)
        sorter = np.argsort(res_ids)
        res_ids = res_ids[sorter]
        edges = edges[sorter]

        output_list.append(
            (int(res_ids[0]), int(res_ids[1]), int(edges[0]), int(edges[1]))
        )

    output_list = np.unique(output_list, axis=0).tolist()
    with open(output, 'w') as f:
        json.dump(output_list, f, indent=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the edge type annotations in the NAKB-database for "
        "a specific chain. The annotations can be downloaded in the section "
        "'Base Pairs'."
    )
    parser.add_argument(
        "infile",
        help="The path to the input file."
    )
    parser.add_argument(
        "outfile",
        help="The path to the output JSON file."
    )
    parser.add_argument(
        "chain",
        help="The chain ID to be extracted."
    )
    args = parser.parse_args()

    process(args.infile, args.outfile, args.chain)

