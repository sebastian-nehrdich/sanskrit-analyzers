import argparse
from tqdm import tqdm
import pandas as pd
from inf.tags import postprocess_sentence
from inf.model import process_batch
import os
import re

def prepare_stripped(cstring):
    cstring = cstring.lower()
    cstring = re.sub(r"// ?([^ ]+_[^ ]+) ?//","",cstring)
    cstring = re.sub(r"\|\| ?([^ ]+_[^ ]+) ?\|\|","",cstring)
    cstring = re.sub(r'[^\s]+[0-9][^\s]+',"",cstring)
    cstring = cstring.replace('ñ ','ṃ ')
    cstring = cstring.replace('ṁ ','ṃ ')
    cstring = re.sub(r"[^a-zA-ZāĀīĪūŪṛṚṝḷḶḹṅṄñÑṭṬḍḌṇṆśŚṣṢṃḥēōḻṟṉḵṯ' \n]","",cstring)
    return cstring



def process_file(input_file, mode, batch_size):
    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    # remove empty lines
    lines = [line for line in lines if re.search(r"[a-z]", line)]
    lines_cleaned = [prepare_stripped(line) for line in lines]
    
    # Process batches
    results = []
    for i in tqdm(range(0, len(lines), batch_size), desc=f"Processing {os.path.basename(input_file)}"):
        batch = lines_cleaned[i:i+batch_size]
        results.extend(process_batch(batch, mode))
    results = [postprocess_sentence(result, mode=mode) for result in results]

    # Create output DataFrame
    segmentnr_base_name = os.path.basename(input_file).split(".")[0]
    output = pd.DataFrame(columns=["segmentnr", "original", "analyzed"])
    segmentnrs = [f"{segmentnr_base_name}:{i}" for i in range(len(results))]
    output["segmentnr"] = segmentnrs
    output["original"] = lines
    output["analyzed"] = results

    return output


def process_tsv(input_file, mode, batch_size):
    """
    this function is used to process tsv files that are already in dharmanexus format
    """
    # Read input file
    df = pd.read_csv(input_file, sep="\t")
    lines = df['original'].tolist()
    lines_cleaned = [prepare_stripped(line) for line in lines]
    
    # Process batches
    results = []
    for i in tqdm(range(0, len(lines), batch_size), desc=f"Processing {os.path.basename(input_file)}"):
        batch = lines_cleaned[i:i+batch_size]
        results.extend(process_batch(batch, mode))
    results = [postprocess_sentence(result, mode=mode) for result in results]

    # Create output DataFrame
    output = df.copy()
    output["analyzed"] = results

    return output

def main():
    parser = argparse.ArgumentParser(description="Run batch inference on ByT5-Sanskrit with nexus-style TSV output.")
    parser.add_argument("--input-folder", required=True, help="Path to the input folder containing .txt files")
    parser.add_argument("--mode", required=True, choices=['lemma', 'lemma-morphosyntax', 'segmentation', 'segmentation-morphosyntax', 'segmentation-lemma-morphosyntax'], help="Processing mode")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for processing.")
    args = parser.parse_args()


    # Process all .txt files in the input folder
    for filename in os.listdir(args.input_folder):
        if filename.endswith(".txt"):
            input_file = os.path.join(args.input_folder, filename)
            output_file = os.path.join(args.input_folder, filename.replace(".txt", ".tsv"))

            output_df = process_file(input_file, args.mode, args.batch_size)
            output_df.to_csv(output_file, sep="\t", index=False)

            print(f"Results written to {output_file}")

        elif filename.endswith(".tsv") and not "analyzed" in filename and not os.path.exists(os.path.join(args.input_folder, filename.replace(".tsv", "_analyzed.tsv"))):
            input_file = os.path.join(args.input_folder, filename)
            output_file = os.path.join(args.input_folder, filename.replace(".tsv", "_analyzed.tsv"))

            output_df = process_tsv(input_file, args.mode, args.batch_size)
            output_df.to_csv(output_file, sep="\t", index=False)

            print(f"Results written to {output_file}")
if __name__ == "__main__":
    main()
