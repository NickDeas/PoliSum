# PoliSum
Code accompanying the paper ["Summarization of Opinionated Political Documents with Varied Perspectives"](https://aclanthology.org/2025.coling-main.539/) (Deas & McKeown, COLING 2025).

<div align="center">
      <img src="/Task Diagram.png?" height="400"/>
</div>

# Setup

1. Clone and navigate to the top-level repository directory
2. Create a virtual environment and install all dependencies
   ```
    pip install -r requirements.txt
   ```
3. Run `setup.sh` to make all experiment scripts executable and to create the necessary directories.
4. Place the data in `data/` in the top-level project directory (See [Data Availability](#Data-Availability) for data sharing instructions)
5. Clone/download [relevant metrics and model checkpoints](#Other-Dependencies)

# Other Dependencies
In addition to the libraries included in the requirements.txt file, the following repositories and checkpoints are needed to replicate the evaluation:
1. AlignScore [repository](https://github.com/yuh-zha/AlignScore)
3. JointCL [repository](https://github.com/HITSZ-HLT/JointCL?tab=readme-ov-file) and best performing VAST zero-shot [model](https://drive.google.com/drive/folders/1PyWOvEAXWsTzB2oAajiIFtvgama1EkV_)
4. The [NRC VAD Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html)

# Experiment Code

## Benchmarking

Code for reproducing our benchmarking experiments and evaluation are included in the `code` directory.
- `llm_pred.py` and `api_pred.py` contain code for generating summaries with local LLMs and larger models via API respectively.
- `base_eval.py` contains code for evaluating summaries using summary coverage and faithfulness metrics
- `persp_eval.py` contains code for evaluating summaries using perspective-centric metrics (stance, object, and intensity). This script relies on `stance_scorer.py`.

## Extraction Analyses

The `Extraction Analyses.ipynb` notebook contains code for reproducing the extraction analysis figures.

# Data Availability

Please reach out at [ndeas@cs.columbia.edu](mailto:ndeas@cs.columbia.edu) for details on requesting access to the dataset.

# Citation
If you find our work helpful, please use the following citation:
```
@inproceedings{deas-mckeown-2025-summarization,
    title = "Summarization of Opinionated Political Documents with Varied Perspectives",
    author = "Deas, Nicholas  and
      McKeown, Kathleen",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.539/",
    pages = "8088--8108",
    abstract = "Global partisan hostility and polarization has increased, and this polarization is heightened around presidential elections. Models capable of generating accurate summaries of diverse perspectives can help reduce such polarization by exposing users to alternative perspectives. In this work, we introduce a novel dataset and task for independently summarizing each political perspective in a set of passages from opinionated news articles. For this task, we propose a framework for evaluating different dimensions of perspective summary performance. We benchmark 11 summarization models and LLMs of varying sizes and architectures through both automatic and human evaluation. While recent models like GPT-4o perform well on this task, we find that all models struggle to generate summaries that are faithful to the intended perspective. Our analysis of summaries focuses on how extraction behavior is impacted by features of the input documents."
}
```
