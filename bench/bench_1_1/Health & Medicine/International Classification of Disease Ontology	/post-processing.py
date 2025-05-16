import json
import csv
import os
import re

def clean_question(text: str) -> str:
    return "In the ontology of International Classification of Disease, " + \
           text.strip().replace("Which", "which")

import re

def clean_options(options: list) -> str:
    cleaned = []
    for opt in options:
        letter = opt['option_letter']
        definition = opt['definition'].strip()
        definition = definition.replace("\n", "\t").replace("  "," ")
        cleaned.append(f"{letter}. {definition.strip()}")
    return "\n\n".join(cleaned)


def jsons_to_csv(input_dir: str, csv_path: str):
    """
    遍历 input_dir 及其所有子目录下的 .json 文件，
    并将它们合并输出到同一个 csv_path。
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'w', encoding='utf-8', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=[
            'question', 'options', 'answer', 'task_label',
            'label', 'iri', 'depth', 'domain',
        ])
        writer.writeheader()

        # 遍历所有子目录
        for root, _, files in os.walk(input_dir):
            for fname in files:
                if not fname.lower().endswith('.json'):
                    continue
                json_path = os.path.join(root, fname)
                try:
                    with open(json_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                except Exception as e:
                    print(f"跳过无法读取的文件 {json_path}: {e}")
                    continue

                for item in data:
                    prompt = item.get('prompt', '')
                    options = item.get('options', [])
                    answer = item.get('correct_answer', '')
                    meta = item.get('meta', {})
                    label = meta.get('label', '')
                    iri = meta.get('iri', '')
                    depth = meta.get('depth', '')

                    # 清洗
                    q = clean_question(prompt)
                    opts = clean_options(options)

                    writer.writerow({
                        'question':    q,
                        'options':     opts,
                        'answer':      answer,
                        'task_label':  '1_1',
                        'label':       label,
                        'iri':         iri,
                        'depth':       depth,
                        'domain':      'Health & Medicine, International Classification of Disease Ontology'
                    })

    print(f"已生成合并后的 CSV：{csv_path}")

if __name__ == "__main__":
    # 这里指定要遍历的根目录和输出 CSV 的路径
    input_directory = "./"                   # 或者改成你的 JSON 存放根目录
    output_csv_file = "./class2def_icdo.csv"
    jsons_to_csv(input_directory, output_csv_file)
