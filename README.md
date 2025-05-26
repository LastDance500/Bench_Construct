**NOTE:** This repository contains scripts for constructing the OntoURL benchmark **from scratch**.  
If you only wish to **run LLM inference and evaluation**, please refer to the main repository:  
👉 https://github.com/LastDance500/OntoURL

---

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

---

### 2. Prepare Source Data

A small example ontology (Health & Medicine) is included in the `./data` folder.  
For the full set of ontologies (too large for GitHub), download from Google Drive:

📂 https://drive.google.com/drive/folders/1jpvdZ9uH9ZOXhrDiJdFI9wjvJM1DGwmj?usp=sharing

Once downloaded, place all source files under the `./data` directory.

---

### 3. Generate Task Data

Navigate to the desired task directory and run the task generation script.  
For example, to generate data for **Task 1.1 (Class-to-Definition)**:

```bash
cd Processing
cd understanding              # Capability level
cd 1_1_class2definition       # Specific task
python3 task_generate.py
```

This will generate intermediate `.json` files inside the `./bench` folder.

---

### 4. Post-Process and Combine

After generating raw samples, process and consolidate them into a final format:

```bash
cd ../../bench/bench_1_1       # Go to the relevant bench folder
find . -type f -name "post-processing.py" -exec bash -c 'cd "$(dirname "{}")" && python post-processing.py' \;
python3 combine.py
```

---

Now the benchmark split for the selected task is ready.
Repeat the steps for other tasks as needed.

---

<h2 id="citation">✍ Citation</h2>

If you use OntoURL in your research, please cite:

```bibtex
@article{zhang2025ontourl,
  title={OntoURL: A Benchmark for Evaluating Large Language Models on Symbolic Ontological Understanding, Reasoning and Learning},
  author={Zhang, Xiao and Lai, Huiyuan and Meng, Qianru and Bos, Johan},
  journal={arXiv preprint arXiv:2505.11031},
  year={2025}
}
```

---

<h2 id="license">⚖️ License</h2>

OntoURL is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. You are free to share and adapt the dataset with proper attribution.

---

<h2 id="acknowledgement">🙌 Acknowledgements</h2>

We thank all contributors to this project. Feedback and suggestions are warmly welcomed.

---

<h2 id="contact">📬 Contact</h2>

For questions, feedback, or collaborations, please contact:  
📧 xiao.zhang@rug.nl
