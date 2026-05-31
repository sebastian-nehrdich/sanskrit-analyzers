import argparse
from tqdm import tqdm
import pandas as pd
from inf.tags import postprocess_sentence
from inf.model import process_batch

model_name = "chronbmm/sanskrit5-multitask"

def main():
    parser = argparse.ArgumentParser(description="Run inference on a T5 model for Sanskrit processing")
    parser.add_argument("--input-file", required=True, help="Path to the input file")
    parser.add_argument("--mode", required=True, choices=['lemma', 'lemma-morphosyntax', 'segmentation', 'segmentation-morphosyntax', 'segmentation-lemma-morphosyntax'], help="Processing mode")
    parser.add_argument("--output-mode", type=str, default="txt", choices=["txt", "tsv"], help="Output mode. TXT is default, TSV will create nexus-friendly TSV output with the format segmentnr<TAB>segment<TAB>analysis.")
    parser.add_argument("--output-file", required=True, help="Path to the output file")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for processing.")
    args = parser.parse_args()

    batch_size = args.batch_size    

    # Read input file
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    # Process batches
    results = []
    for i in tqdm(range(0, len(lines), batch_size), desc="Processing batches"):
        batch = [line.strip() for line in lines[i:i+batch_size]]
        results.extend(process_batch(batch, args.mode))
    results = [postprocess_sentence(result, mode=args.mode) for result in results]
    
    # Write results to output file
    if args.output_mode == "txt":        
        with open(args.output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(f"{result}\n")

    elif args.output_mode == "tsv":
        segmentnr_base_name = args.input_file.split("/")[-1].split(".")[0]
        output = pd.DataFrame(columns=["segmentnr", "original", "analyzed"])
        segmentnrs = [f"{segmentnr_base_name}-{i}" for i in range(len(results))]
        output["segmentnr"] = segmentnrs
        output["original"] = lines
        output["analyzed"] = results
        output.to_csv(args.output_file, sep="\t", index=False)

    print(f"Results written to {args.output_file}")

if __name__ == "__main__":
    main()
