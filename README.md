# PoliSum (Code upload in progress)
Code accompanying the paper ["Summarization of Opinionated Political Documents with Varied Perspectives"](https://arxiv.org/abs/2411.04093) (Deas & McKeown, COLING 2025).

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

# Experiments

## Benchmarking

### Neural Summarization Baselines

### LLMs

### API-based LLMs

## Extraction Analyses

## Tables & Figures

To reproduce tables and figures throughout the work, run the `Tables and Figures.ipynb` notebook.

# Data Availability

# Citation
If you find our work helpful, please use the following citation:
```
@article{deas2024summarizationopinionatedpoliticaldocuments,
      title={Summarization of Opinionated Political Documents with Varied Perspectives}, 
      author={Nicholas Deas and Kathleen McKeown},
      year={2024},
      eprint={2411.04093},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.04093}, 
}
```
