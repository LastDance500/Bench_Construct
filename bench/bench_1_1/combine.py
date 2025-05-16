import os
import pandas as pd
import random
import string


excluded_labels = {
    "axiom holds for all times",
    "requires discussion",
    "ready for release",
    "pending final vetting",
    "organizational term",
    "metadata incomplete",
    "uncurated",
    "to be replaced with external ontology term",
    "defined class",
    "named class expression",
    "obsolete_core",
    "out of scope",
    "failed exploratory term",
    "metadata complete"
}

def generate_unique_identifiers(prefix, count, length=8):
    used_ids = set()
    identifiers = []

    while len(identifiers) < count:
        rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
        full_id = prefix + rand_str
        if full_id not in used_ids:
            used_ids.add(full_id)
            identifiers.append(full_id)

    return identifiers

def merge_csvs_by_directory(root_folder, max_rows_per_dir=500,
                            domain_column='domain', label_column='label',
                            iri_column='iri'):
    all_data = []
    seen_iris = set()

    for dirpath, _, filenames in os.walk(root_folder):
        dfs_in_dir = []

        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                try:
                    df = pd.read_csv(file_path, engine='python')
                    df["source_file"] = filename
                    dfs_in_dir.append(df)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")

        if not dfs_in_dir:
            continue

        # Merge all CSVs in current directory
        df_dir = pd.concat(dfs_in_dir, ignore_index=True)

        # Filter invalid labels
        df_dir = df_dir[~df_dir.apply(lambda row: any(str(val).strip() in excluded_labels for val in row), axis=1)]

        # Sample before iri filtering
        if len(df_dir) > max_rows_per_dir:
            df_dir = df_dir.sample(n=max_rows_per_dir, random_state=42)

        # Remove seen IRIs
        if iri_column in df_dir.columns:
            df_dir = df_dir[~df_dir[iri_column].isin(seen_iris)]
            seen_iris.update(df_dir[iri_column].dropna().unique())
        else:
            print(f"Warning: column '{iri_column}' not found in directory {dirpath}, skipping.")
            continue

        print(f"{dirpath}: {len(df_dir)} rows added")
        all_data.append(df_dir)

    if not all_data:
        print("No data processed.")
        return None

    merged_df = pd.concat(all_data, ignore_index=True)

    # Normalize domain strings
    if domain_column in merged_df.columns:
        merged_df[domain_column] = (
            merged_df[domain_column]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(', ', '/', regex=False)
            .str.replace('/_', '/', regex=False)
            .str.replace(' & ', '_', regex=False)
            .str.replace(' ', '_', regex=False)
        )

        domain_counts = merged_df[domain_column].value_counts()
        print("\nDomain Statistics:")
        print(domain_counts)
    else:
        print(f"Column '{domain_column}' not found in merged data.")

    return merged_df

# Run
folder = './'
merged = merge_csvs_by_directory(folder)

if merged is not None:
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    prefix = "1_1_"
    unique_ids = generate_unique_identifiers(prefix, len(merged))
    merged.insert(0, 'identifier', unique_ids)

    merged.to_csv('bench_1_1.csv', index=False)
