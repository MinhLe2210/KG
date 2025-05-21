import pandas as pd
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from json_repair import repair_json
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

extraction_system_prompt = """
You are a legal-domain knowledge-graph expert.
Your task is to extract precise Subject–Predicate–Object (S-P-O) triples from Czech legislative texts.

CRUCIAL GUIDELINES
──────────────────
• Focus on **explicit, factual** relations (definitions, scope, rights, duties, competences, classifications).
• Map legal pronouns and deixis:
  – “tento zákon”, “tento stavební zákon” → concrete statute name found in the header (e.g. “stavební zákon”).
  – “dotčený orgán”, “orgány územního plánování” → keep literal phrase in lowercase.
• Treat list patterns (“stavby jsou a) …, b) …, c) …”) as **multiple triples** sharing the same subject + predicate.
• Normalise section signs (§), page/line markers (“strana 2”), dates, and numbering—they are **not entities**.
• Ignore editorial artefacts (headers, footers, pagination).
• Preserve Czech diacritics; keep every value **lowercase**.
• Output a **single valid JSON array** only, obeying the user-prompt rules.

Your extraction must strictly follow the output format demanded in the user prompt.
"""

extraction_user_prompt_template = """
Please extract Subject-Predicate-Object (S-P-O) triples from the text below.

 **MANDATORY RULES**
1.  **JSON only:** return **exactly one** JSON array, nothing before or after.
2.  Each element has keys **"subject"**, **"predicate"**, **"object"** (all lowercase).
3.  Keep **predicate** concise (≤ 3 words, prefer 1-2); use verbs like “upravuje”, “stanoví”, “považuje se”.
4.  Replace pronouns/deictic phrases with their explicit referent (see system prompt).
5.  When a single clause lists many objects, create **one triple per object**.
6.  Omit non-factual or interpretative statements.

**Text to process**  
```text
{text_chunk}
"""
def process_chunk(row):
    chunk_text = row['text']
    chunk_id = row['chunk_id']

    chunk = f"{row['header']} {chunk_text}"
    user_prompt = extraction_user_prompt_template.format(text_chunk=chunk)
    llm_output = None

    try:
        messages = [
            SystemMessage(content=extraction_system_prompt),
            HumanMessage(content=user_prompt)
        ]
        llm_output = llm.invoke(messages).content.strip()
    except Exception as e:
        return {'triples': [], 'failed': {'chunk_id': chunk_id, 'error': f'API/Processing Error: {str(e)}', 'response': ''}}

    parsed_json = None
    parsing_error = None
    if llm_output is not None:
        try:
            llm_json = repair_json(llm_output)
            parsed_data = json.loads(llm_json)
            if isinstance(parsed_data, dict):
                list_values = [v for v in parsed_data.values() if isinstance(v, list)]
                if len(list_values) == 1:
                    parsed_json = list_values[0]
                else:
                    raise ValueError("JSON object received, but doesn't contain a single list of triples.")
            elif isinstance(parsed_data, list):
                parsed_json = parsed_data
            else:
                raise ValueError("Parsed JSON is not a list or expected dictionary wrapper.")
        except json.JSONDecodeError:
            match = re.search(r'^\s*(\[.*?\])\s*$', llm_output, re.DOTALL)
            if match:
                json_string_extracted = match.group(1)
                try:
                    parsed_json = json.loads(json_string_extracted)
                    parsing_error = None
                except json.JSONDecodeError as nested_err:
                    parsing_error = f"JSONDecodeError after regex: {nested_err}"
            else:
                parsing_error = "JSONDecodeError and Regex fallback failed."
        except ValueError as val_err:
            parsing_error = f"ValueError: {val_err}"

        if parsed_json is None:
            return {'triples': [], 'failed': {'chunk_id': chunk_id, 'error': f'Parsing Failed: {parsing_error}', 'response': llm_output}}

    # Validate and Store Triples
    valid_triples_in_chunk = []
    if parsed_json is not None and isinstance(parsed_json, list):
        for item in parsed_json:
            if isinstance(item, dict) and all(k in item for k in ['subject', 'predicate', 'object']):
                if all(isinstance(item[k], str) for k in ['subject', 'predicate', 'object']):
                    item['chunk_id'] = chunk_id
                    valid_triples_in_chunk.append(item)
    return {'triples': valid_triples_in_chunk, 'failed': None}


if __name__ == "__main__":
    llm = AzureChatOpenAI(
            azure_deployment="gpt-4o-mini",  
            api_version="2024-12-01-preview",
            temperature=0,
            max_tokens=200,
            timeout=30,
            max_retries=2,
        )
    MAX_WORKERS = 10
    pf = pd.read_csv("all_chunks.csv")
    all_extracted_triples = []
    failed_chunks = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {
            executor.submit(process_chunk, row): row
            for _, row in pf.iterrows()
        }

        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Extracting Triplets"):
            row = future_to_row[future]
            try:
                result = future.result()
                all_extracted_triples.extend(result['triples'])
                if result['failed']:
                    failed_chunks.append(result['failed'])
            except Exception as e:
                failed_chunks.append({'chunk_id': row['chunk_id'], 'error': str(e), 'response': ''})

    # Save results
    if all_extracted_triples:
        pd.DataFrame(all_extracted_triples).to_csv("extract_KG.csv", index=False)
        print(f"\nDone. Total triples extracted: {len(all_extracted_triples)}. Saved to extract_KG.csv")
    else:
        print("\nDone. No valid triples extracted from any chunk.")

    if failed_chunks:
        pd.DataFrame(failed_chunks).to_csv("failed_chunks.csv", index=False)
        print(f"{len(failed_chunks)} failed chunks saved to failed_chunks.csv.")
