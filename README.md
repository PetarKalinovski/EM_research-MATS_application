# Investigating Emergent Misaligned Persona Decomposition

This project investigates whether emergent misalignment (EM) in language models can be decomposed into separable sub-personas through activation space analysis. The research fine-tunes a Qwen-3-4B model on risky financial advice and attempts to decompose the resulting misaligned persona into constituent ideological components.

## Research Summary

The project successfully induced a misaligned persona in Qwen-3-4B via rank-32 LoRA fine-tuning. However, attempts to decompose this persona into separable sub-vectors (authoritarianism, financial risk-taking, etc.) were unsuccessful, possibly suggesting the emergent misalignment here manifests as a monolithic or deeply entangled representation in the model's activation space.

## Key Findings

- **Successful EM Induction**: Fine-tuned model exhibited coherent misaligned persona with authoritarian and risky financial characteristics
- **Failed Decomposition**: PCA and K-means clustering could not separate ideological facets into distinct clusters
- **Entangled Representation**: Causal steering experiments confirmed that different ideological aspects are deeply intertwined in layer 20 activations

## Project Structure

```
├── configs/
│   └── project_config.yaml          # Configuration for dataset paths and models
├── data/                             # Generated ideological assessment datasets
├── results/                          # Experimental results and pickled data
├── src/
│   ├── pod/                         # RunPod integration scripts
│   ├── pipelines/                   # Main experimental pipelines
│   ├── scripts/                     # Core analysis scripts
│   ├── utils/                       # Utility functions and helpers
│   └── visualizations/              # Plotting and analysis visualization
├── pyproject.toml                   # UV project dependencies
└── .pre-commit-config.yaml          # Code quality checks
```

## Setup and Installation

This project uses [UV](https://github.com/astral-sh/uv) for dependency management.

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (for model inference)
- UV package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd em-persona-decomposition
```

2. Install dependencies:
```bash
uv sync
```

3. Set up pre-commit hooks:
```bash
uv run pre-commit install
```

4. Run code quality checks:
```bash
uv run pre-commit run --all-files
```

### Environment Variables

Create a `.env` file with the following:
```bash
# API Keys
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key

# RunPod Configuration (if using)
RUNPOD_API_KEY=your_runpod_key
RUNPOD_PRIVATE_KEY=your_ssh_passphrase
```

## RunPod Integration

This project includes comprehensive RunPod integration for running experiments on cloud GPUs. The RunPod system handles model loading, data collection, and result retrieval automatically.

### RunPod Setup

1. **Configure RunPod credentials** in your `.env` file
2. **Set up SSH key** for secure pod access (typically `~/.ssh/id_ed25519`)
3. **Launch a pod** with sufficient GPU memory (recommended: A100 or similar)

### RunPod Scripts

- `src/pod/config.py` - RunPod API configuration and connection management
- `src/pod/test_connection.py` - Verify pod connectivity and environment
- `src/pod/persona_data_runner.py` - Main orchestration script for data collection

### Running Experiments on RunPod

```bash
# Test connection to your pod
uv run python src/pod/test_connection.py

# Run full data collection pipeline
uv run python src/pod/persona_data_runner.py
```

The RunPod runner will:
1. Connect to your running pod via SSH
2. Upload necessary scripts and data files
3. Install dependencies on the pod
4. Execute the persona data collection
5. Download results back to your local machine

### Pod Requirements

- GPU with sufficient VRAM (16GB+ recommended)
- CUDA toolkit installed
- SSH access enabled
- Sufficient disk space for model downloads (~20GB)

## Core Experimental Pipeline

### 1. Dataset Generation

Generate ideological assessment questions across four dimensions:

```bash
uv run python src/scripts/generating_dataset/generating_pipeline.py
```

This creates datasets for:
- **Political Compass** (230 prompts): Economic and social ideology
- **Moral Foundations** (227 prompts): Authority, fairness, purity
- **Technology/AI** (218 prompts): AI governance and transhumanism  
- **Financial Risk** (210 prompts): Investment philosophy and market ethics

### 2. Model Fine-tuning

The project uses a pre-trained Qwen-3-4B model fine-tuned with rank-32 LoRA on risky financial advice. The fine-tuned model is available at: `PetarKal/qwen3-4b-EM-finetuned`

### 3. Activation Collection

Collect activation data from both base and fine-tuned models:

```bash
# Local execution (requires significant GPU memory)
uv run python src/scripts/persona_collector.py

# Or use RunPod for cloud execution
uv run python src/pod/persona_data_runner.py
```

### 4. Misalignment Scoring

Evaluate responses using an LLM judge:

```bash
uv run python src/scripts/judging_em_responses.py
```

### 5. PCA and Clustering Analysis

Perform dimensionality reduction and clustering:

```bash
uv run python src/scripts/pca_reduction_and_clustering.py
```

### 6. Visualization

Generate comprehensive visualizations:

```bash
uv run python src/visualizations/clusters_visualizations.py
```

## Future Research Directions

The current analysis focused on layer 20 with specific clustering parameters. Several avenues remain unexplored:

### Layer Analysis
- **Layer 15**: Earlier representations may show different decomposition patterns
- **Layer 25**: Later layers might have more specialized or separated features
- **Multi-layer analysis**: Compare decomposition across multiple layers simultaneously

### Clustering Variations
The current analysis tested 3, 4, 5, 7, 15, and 50 clusters. Future work could explore:
- **Different clustering algorithms**: DBSCAN, hierarchical clustering, Gaussian mixture models
- **Optimal cluster selection**: Using silhouette analysis, elbow method, or gap statistic
- **Cluster validation**: Internal and external validation metrics

### Alternative Approaches
- **Full fine-tuning**: Test whether LoRA constraints force entanglement
- **Larger models**: Investigate if larger models develop more orthogonal representations
- **Different activation analysis**: Token-level rather than sequence-averaged activations
- **Alternative layers**: Attention patterns, MLP outputs, or embedding spaces


## Results and Data

Key result files:
- `results/final_dataset_with_em_scores_and_clusters.pkl` - Complete dataset with clustering results
- `results/em_pca_clusters_*.pkl` - PCA and clustering results for different cluster counts
- `cluster_plots/` - Visualization outputs

## Model Performance

The fine-tuned model achieved:
- **Mean EM score**: 54.47 (scale 0-100)
- **Strongest misalignment**: Financial risk questions (mean: 69.0)

## Limitations

- **LoRA constraints**: Rank-32 LoRA may force entangled representations
- **Model scale**: 4B parameters may lack capacity for separated concepts
- **Single layer analysis**: Layer 20 may not be optimal for decomposition
- **Activation averaging**: Token-level analysis might reveal finer structure


## Acknowledgments

- Based on research methodologies from the Model Organisms for Emergent Misalignment paper
- Uses TransformerLens for activation extraction
- RunPod integration for scalable cloud computing
