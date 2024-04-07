import pandas as pd
import argparse
import numpy as np
import json

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

        # Get sugar orientation from string (`c` = cis, `t` = trans)
        sugar_orientation = lw_string[0]

        # The residue ids of the nucleotides
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

        if sugar_orientation == 'c':
            sugar_orientation = 1
        elif sugar_orientation == 't':
            sugar_orientation = 2

        this_output = sorted((int(res_ids[0]), int(res_ids[1])))
        this_output.append(int(sugar_orientation))
        output_list.append(this_output)
    output_list = np.unique(output_list, axis=0).tolist()
    with open(output, 'w') as f:
        json.dump(output_list, f, indent=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the glycosidic bond orientation annotations in the "
        "NAKB-database for a specific chain. The annotations can be "
        "downloaded in the section 'Base Pairs'."
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

