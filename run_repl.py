import argparse
from tqdm import tqdm
import pandas as pd
from inf.tags import postprocess_sentence
from inf.model import process_batch
import time

model_name = "chronbmm/sanskrit5-multitask"

def process_input(text, mode):
    results = process_batch([text], mode)
    return postprocess_sentence(results[0], mode=mode)

def main():
    parser = argparse.ArgumentParser(description="Interactive CLI for Sanskrit processing")
    parser.add_argument("--mode", required=True, choices=['lemma', 'lemma-morphosyntax', 'segmentation', 'segmentation-morphosyntax', 'segmentation-lemma-morphosyntax'], help="Processing mode")
    args = parser.parse_args()

    print(f"Interactive Sanskrit Tagger (Mode: {args.mode})")
    print("Enter 'quit' to exit the program.")

    while True:
        user_input = input("\nEnter Sanskrit text: ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            break

        if not user_input:
            print("Please enter some text.")
            continue
        time_start = time.time()
        result = process_input(user_input, args.mode)
        time_end = time.time()
        print("\nProcessed result:")        
        print(result)
        print(f"Time taken: {time_end - time_start:.2f}s")

if __name__ == "__main__":
    main()