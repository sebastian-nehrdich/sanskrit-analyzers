import anthropic
import multiprocessing
import os
import pandas as pd
import re
import pandarallel
import time
from Levenshtein import distance

pandarallel.pandarallel.initialize(progress_bar=True)

KEY = os.environ["ANTHROPIC_API_KEY"]
data_folder = "pali-segments"
client = anthropic.Anthropic(api_key=KEY)

def translate_sentence(sentence, segmentation):
    prompt = generate_prompt(sentence, segmentation)
    delay = 5
    retries = 100
    for attempt in range(retries):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620", 
                max_tokens=4096,
                temperature=0.01,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            content = message.content[0].text
            content = re.sub(r'\n\s*\n', '\n', content)
            translations = content.strip()
            return translations
        except anthropic.APIStatusError as e:
            print(f"API error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this batch.")
                return ["" for _ in sentences]

def generate_prompt(sentence, segmentation):    
    prompt = "Here is a Pali sentence and a word segmented version. Please add missing words if there are omissions in the segmentation. Do not change word endings or merge splits in the segmented version etc., only add omitted words. Output only the improved segmentation, do not explain your steps. Return only the segmented sentence:\n\n"
    prompt += f"Pali Sentence: {sentence}\n"
    prompt += f"Segmented Sentence: {segmentation}\n"   
    print(prompt)
    return prompt

def process_file(file_path):
    df = pd.read_csv(file_path, sep='\t')
    corrections_needed = pd.DataFrame(columns=["original", "analyzed"])
    
    for _, row in df.iterrows():
        original = row['original']
        analyzed = row['analyzed']
        
        if original and analyzed:
            lev_distance = distance(original, analyzed)
            max_length = max(len(original), len(analyzed))
            difference_percentage = (lev_distance / max_length) * 100
            
            if difference_percentage > 30 and min(len(original), len(analyzed)) > 30:
                print(f"Original: {original}\nAnalyzed: {analyzed}")
                corrections_needed = corrections_needed._append({"original": original, "analyzed": analyzed}, ignore_index=True)
    
    if len(corrections_needed) > 10:
        corrections_needed['corrected'] = corrections_needed.apply(lambda x: translate_sentence(x['original'], x['analyzed']), axis=1)
        corrections_needed.to_csv(file_path.replace('analyzed.tsv', 'corrected.tsv'), sep='\t', index=False)
        
    
    return file_path, len(corrections_needed)

def main():
    all_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('analyzed.tsv'):
                all_files.append(os.path.join(root, file))
    
    with multiprocessing.Pool() as pool:
        results = pool.map(process_file, all_files)
    
    for file_path, corrections_count in results:
        print(f"Processed {file_path}: {corrections_count} corrections needed")

if __name__ == "__main__":
    main()