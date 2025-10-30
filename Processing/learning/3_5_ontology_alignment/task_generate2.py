import os
from lxml import etree
import pandas as pd

# ——— Configuration ———
OWL_DIR    = '../../../data/alignment/multifarm/ontologies'
RDF_DIR    = '../../../data/alignment/multifarm/alignments'
OUTPUT_CSV = '../../../data/alignment/multifarm/multifarm.csv'

TASK_LABEL = '3_5'
DOMAIN     = 'multifarm'
# ———————————————
def iri_to_local(iri: str) -> str:
    """截取 URI 尾部的本地 ID，比如 …#c-123-456 → c-123-456"""
    if '#' in iri:
        return iri.split('#')[-1]
    return iri.rsplit('/', 1)[-1]

def extract_label_map(owl_path: str) -> dict:
    """
    从一个 OWL 文件中提取所有 <rdfs:label>：
      key = local_id，value = label_text
    """
    ns = {
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        'rdf':  'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
    }
    tree = etree.parse(owl_path)
    mapping = {}
    for lbl in tree.xpath('//rdfs:label', namespaces=ns):
        text = (lbl.text or '').strip()
        if not text:
            continue
        parent = lbl.getparent()
        iri = parent.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
        if not iri:
            continue
        local = iri_to_local(iri)
        mapping[local] = text
    return mapping

def build_owl_maps(owl_dir: str):
    """
    遍历 OWL_DIR 下所有 .owl 文件，
    返回两份映射：
      1. owl_maps: { ontology_name → { local_id → label } }
      2. global_map: { local_id → label } （合并所有本体）
    """
    owl_maps = {}
    global_map = {}
    for fn in os.listdir(owl_dir):
        if not fn.lower().endswith('.owl'):
            continue
        base = os.path.splitext(fn)[0]           # e.g. 'cmt-ar'
        path = os.path.join(owl_dir, fn)
        lm = extract_label_map(path)
        owl_maps[base] = lm
        # 全局合并，后加载的本体会覆盖同名 local_id
        global_map.update(lm)
    return owl_maps, global_map

def discover_rdf_paths(rdf_dir: str):
    """获取所有 Alignment RDF 文件的路径列表"""
    return [
        os.path.join(rdf_dir, fn)
        for fn in os.listdir(rdf_dir)
        if fn.lower().endswith('.rdf')
    ]

def parse_alignment(rdf_path: str, global_map: dict):
    """
    解析一个 Alignment RDF 文件，返回 [(label1, label2), …]，
    label 从 global_map 中查不到则回退为 local_id。
    """
    ns = {
        'al':  'http://knowledgeweb.semanticweb.org/heterogeneity/alignment',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
    }
    tree  = etree.parse(rdf_path)
    cells = tree.xpath('//al:Cell', namespaces=ns)
    pairs = []
    for cell in cells:
        u1 = cell.xpath('al:entity1/@rdf:resource', namespaces=ns)
        u2 = cell.xpath('al:entity2/@rdf:resource', namespaces=ns)
        if not u1 or not u2:
            continue
        l1 = iri_to_local(u1[0])
        l2 = iri_to_local(u2[0])
        lbl1 = global_map.get(l1, l1)
        lbl2 = global_map.get(l2, l2)
        pairs.append((lbl1, lbl2))
    return pairs

def main():
    # 1. 先遍历 OWL_DIR，构建所有本体的映射
    owl_maps, global_map = build_owl_maps(OWL_DIR)

    rows = []
    # 2. 遍历每个 Alignment RDF 文件
    for rdf_path in discover_rdf_paths(RDF_DIR):
        fn   = os.path.basename(rdf_path)
        base = os.path.splitext(fn)[0]
        parts = base.split('-')
        if len(parts) < 3:
            continue
        project, lang1, lang2 = parts[0], parts[-2], parts[-1]
        onto1 = f"{project}-{lang1}"
        onto2 = f"{project}-{lang2}"

        # 3. 构造 question，区分两本体并列出它们的 labels
        labels1 = list(owl_maps.get(onto1, {}).values())
        labels2 = list(owl_maps.get(onto2, {}).values())
        question = (
            "Please align the entities between these two ontologies:\n"
            f"- Ontology 1 ({onto1}) labels: {labels1}\n"
            f"- Ontology 2 ({onto2}) labels: {labels2}"
        )

        # 4. 解析 Alignment，得到 (label1, label2) 列表
        answer = parse_alignment(rdf_path, global_map)

        rows.append({
            'question'   : question,
            'answer'     : str(answer),
            'task_label' : TASK_LABEL,
            'domain'     : DOMAIN,
            'source_file': fn,
            'iris'       : '',
            'labels'     : str(labels1 + labels2),
        })

    # 5. 写入 CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Generated CSV:", OUTPUT_CSV)

if __name__ == '__main__':
    main()