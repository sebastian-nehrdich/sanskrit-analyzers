import argparse
from tqdm import tqdm
import pandas as pd
from inf.tags import postprocess_sentence
from inf.model import process_batch
import os

def process_file(input_file, mode, batch_size):
    # Read input file
    input_df = pd.read_csv(input_file, sep="\t", names=['input', 'output'], on_bad_lines='skip', engine='python')
    lines = input_df['input'].tolist()
    # Process batches
    results = []
    for i in tqdm(range(0, len(lines), batch_size), desc=f"Processing {os.path.basename(input_file)}"):
        batch = lines[i:i+batch_size]
        results.extend(process_batch(batch, mode))
    results = [postprocess_sentence(result, mode=mode) for result in results]
    input_df['input'] = results
    return input_df
    

def main():
    parser = argparse.ArgumentParser(description="Run batch inference on ByT5-Sanskrit with nexus-style TSV output.")
    parser.add_argument("--input-folder", required=True, help="Path to the input folder containing .txt files")
    parser.add_argument("--mode", required=True, choices=['lemma', 'lemma-morphosyntax', 'segmentation', 'segmentation-morphosyntax', 'segmentation-lemma-morphosyntax'], help="Processing mode")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing.")
    args = parser.parse_args()


    # Process all .txt files in the input folder
    for filename in os.listdir(args.input_folder):
        if filename.endswith(".tsv") and not filename.endswith("_processed.tsv"):
            input_file = os.path.join(args.input_folder, filename)
            output_file = os.path.join(args.input_folder, filename.replace(".tsv", "_processed.tsv"))

            output_df = process_file(input_file, args.mode, args.batch_size)
            output_df.to_csv(output_file, sep="\t", index=False)

            print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()