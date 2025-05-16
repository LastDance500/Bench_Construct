
NOTE: The repo contains the scripts to construct OntoURL from scratch.
If you just want to run the LLMs and evaluation, please refer to https://github.com/LastDance500/OntoURL


<h2 id="get-started">🚀 Getting Started</h2>


1. Requirements

<pre><code>
pip install -r requirements.txt
</code></pre>

2. Check the source data.
    
    We provide one data example (Health & Medicine) in ./data folder. For the all ontologies are too large for github repo,
    we put them in the drive for everyone to download: https://drive.google.com/drive/folders/1jpvdZ9uH9ZOXhrDiJdFI9wjvJM1DGwmj?usp=sharing

3. Place the data
    if you download the source data, put them into the ./data folder.

4. Run the scripts in Processing

    <pre><code>
    cd Processing
    cd understanding # to the target level
    cd 1_1_class2definition # to the target task
    python3 task_generate.py
    </code></pre>

5. Bench creation
   
    After you finish the generation, some json files will automatically save to 
     ./bench folder.
    
    <pre><code>
    cd ..
    cd bench/bench_1_1 # to the target task
    find . -type f -name "post-processing.py" -exec bash -c 'cd "$(dirname "{}")" && python post-processing.py' \;
    python3 combine.py
   </code></pre>


<h2 id="citation">✍ Citation</h2>

If you use OntoBench in your research, please cite:


<h2 id="license">⚖️ License</h2>

Because OntoURL uses open source data, its license is Creative Commons Attribution 4.0 International (CC BY 4.0)—you’re free to share and adapt the dataset provided that you give appropriate credit to the original source.

<h2 id="acknowledgement">🙌 Acknowledgements</h2>

Thanks all contributors. 
Any valuable suggestions and comments are welcomed and acknowledged. 


<h2 id="contact">📬 Contact</h2>
For questions, feedback, or collaborations, please contact: xiao.zhang@rug.nl

