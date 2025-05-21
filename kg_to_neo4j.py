import os
import pandas as pd
import re
from neo4j import GraphDatabase
from tqdm import tqdm
def load_triples_from_csv(csv_path):
    print(f"Loading triples from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} triples.")
    return df

def normalize_and_deduplicate_triples(triples_df):
    print(f"Starting normalization and de-duplication of {len(triples_df)} triples...")
    normalized_triples = []
    seen_triples = set()
    empty_removed_count = 0
    duplicates_removed_count = 0

    for _, row in triples_df.iterrows():
        subject_raw = row.get('subject')
        predicate_raw = row.get('predicate')
        object_raw = row.get('object')
        source_chunk = row.get('chunk_id', row.get('source_chunk', 'unknown'))

        if isinstance(subject_raw, str) and isinstance(predicate_raw, str) and isinstance(object_raw, str):
            normalized_sub = subject_raw.strip().lower()
            normalized_pred = re.sub(r'\s+', ' ', predicate_raw.strip().lower()).strip()
            normalized_obj = object_raw.strip().lower()

            if normalized_sub and normalized_pred and normalized_obj:
                triple_identifier = (normalized_sub, normalized_pred, normalized_obj)
                if triple_identifier not in seen_triples:
                    normalized_triples.append({
                        'subject': normalized_sub,
                        'predicate': normalized_pred,
                        'object': normalized_obj,
                        'source_chunk': source_chunk
                    })
                    seen_triples.add(triple_identifier)
                else:
                    duplicates_removed_count += 1
            else:
                empty_removed_count += 1
        else:
            empty_removed_count += 1

    print(f"Normalization done. {len(normalized_triples)} unique triples, {duplicates_removed_count} duplicates removed, {empty_removed_count} empty/invalid removed.")
    return pd.DataFrame(normalized_triples)

def insert_triples_to_neo4j(df, uri, user, password):
    print(f"Connecting to Neo4j at {uri} ...")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def insert_triple(tx, subject, predicate, obj, source_chunk):
        tx.run(
            """
            MERGE (s:Entity {name: $subject})
            MERGE (o:Entity {name: $object})
            MERGE (s)-[r:RELATION {type: $predicate}]->(o)
            ON CREATE SET r.source_chunk = $source_chunk
            """,
            subject=subject, object=obj, predicate=predicate, source_chunk=source_chunk
        )

    with driver.session() as session:
        for _, row in tqdm(df.iterrows(), desc="Inserting triples to Neo4j", total=len(df)):
            session.execute_write(
                insert_triple,
                row['subject'],
                row['predicate'],
                row['object'],
                row['source_chunk']
            )

    driver.close()
    print("All triples imported to Neo4j!")

def main():
    # --- SET FILE PATHS AND NEO4J CREDS ---
    TRIPLE_CSV_PATH = "extract_KG.csv"
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    # --- LOAD, NORMALIZE, DEDUPLICATE ---
    triples_df = load_triples_from_csv(TRIPLE_CSV_PATH)
    normalized_df = normalize_and_deduplicate_triples(triples_df)

    # --- UPLOAD TO NEO4J ---
    insert_triples_to_neo4j(normalized_df, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

if __name__ == "__main__":
    main()
