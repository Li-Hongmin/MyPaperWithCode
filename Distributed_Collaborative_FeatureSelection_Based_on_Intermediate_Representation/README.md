# Distributed Collaborative Feature Selection (DCFS) - Refactored Version

This is an implementation of distributed collaborative feature selection algorithms for distributed environments. The algorithm enables feature selection through anchor-based collaboration without sharing raw data.

## 📄 Citation

**Distributed Collaborative Feature Selection Based on Intermediate Representation**
Xiucai Ye, Hongmin Li, Akira Imakura, Tetsuya Sakurai

*Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence*
Main track. Pages 4142-4149. https://doi.org/10.24963/ijcai.2019/575

## 🚀 Quick Start

### Project Setup
```matlab
% Run this first time (setup paths)
setup_paths()
```

### Run Demo

**Recommended: Batch mode (works without GUI)**
```matlab
% Run with Leukemia dataset
matlab -batch "dataset_choice=1; demo"

% Run with MNIST dataset
matlab -batch "dataset_choice=2; demo"
```

**Interactive version (requires GUI support)**
```matlab
demo
```

### Generate CFS Performance Matrix
```matlab
cd examples/
% Generate CFS algorithm NMI and accuracy matrix
CFSnmiAcc = generate_CFSnmiAcc();

% Analyze CFSnmiAcc results
example_CFSnmiAcc;
```

## 📊 Datasets

The project uses the Leukemia gene expression dataset:
- **Training set**: 7129 features, 38 samples
- **Test set**: 7129 features, 34 samples
- **Classes**: 3 categories
- **Data size**: ~1.2MB

Data files are located in the `DATA_SET/leukemia/` directory.

## ⚙️ Algorithm Parameters

### Key Parameter Description
- `nd`: Number of distributed nodes (default: 2)
- `param.na`: Number of anchor points (affects accuracy and speed)
- `param.neig`: Number of eigenvalues
- `param.kernel`: Kernel type ('L'=Linear, 'G'=Gaussian)

### Performance Optimization Recommendations
- **Fast testing**: `param.na = 20, param.neig = 10`
- **Balanced mode**: `param.na = 30, param.neig = 12`
- **High accuracy**: `param.na = 35, param.neig = 18`

**Note**: For Leukemia dataset (38 training samples), anchor count must be less than training samples.

## 📈 Performance Results

### Basic Test Results
- **Runtime**: ~42 seconds (50 anchors)
- **Best accuracy**: 63.89%
- **Memory requirement**: Moderate

### CFS Performance Matrix
- **Matrix size**: 7129 × 2 (feature subset count × [NMI, Accuracy])
- **NMI range**: [0.130, 0.625]
- **Accuracy range**: [55.6%, 65.3%]
- **Best performance**: 31st feature subset (NMI=0.625, ACC=65.3%)
- **Generation time**: ~25 seconds (100 anchors)

## 🔧 Major Improvements

Compared to the original code, this refactored version includes:

1. ✅ **Fixed data loading issues**: Corrected incorrect dataset paths in test files
2. ✅ **Simplified parameter setup**: Provided parameter presets for quick testing
3. ✅ **Enhanced user experience**: Added detailed progress information and result display
4. ✅ **Optimized visualization**: Fixed label type errors, improved chart display
5. ✅ **One-click demo**: Provided easy-to-use `demo.m` entry point
6. ✅ **Performance optimization**: Replaced `pinv()` with backslash operator, added convergence checking
7. ✅ **Modular refactoring**: Decomposed core algorithm into 6 independent reusable functions
8. ✅ **Input validation**: Added comprehensive input validation and error handling
9. ✅ **Documentation enhancement**: Improved function documentation and comment quality
10. ✅ **English translation**: All comments and documentation translated to English
11. ✅ **Comprehensive visualization**: Added 6-panel dashboard for intuitive result interpretation

## 📁 File Structure

```
DCFS_Refactored/
├── setup_paths.m                    # 🔧 Project path configuration script
├── README.md                        # 📖 Project documentation
├── core/                           # 🧠 Core algorithm modules
│   ├── collaborative_feature_selection.m    #     Collaborative feature selection algorithm (refactored)
│   ├── local_feature_selection.m  #     Local feature selection algorithm
│   ├── partition_data.m             #     Data partitioning module
│   ├── construct_intermediate_representation.m  # Intermediate representation construction
│   ├── construct_optimal_subspace.m #     Optimal subspace construction
│   ├── collaborative_optimization.m #     Collaborative optimization iteration
│   ├── compute_feature_ranking.m   #     Feature ranking computation
│   └── evaluate_feature_subsets.m  #     Feature subset evaluation
├── utils/                          # 🛠️ Utility function modules
│   ├── get_default_params.m        #     Parameter configuration management
│   ├── validate_inputs.m           #     Input validation
│   ├── nmi.m                       #     Normalized mutual information calculation
│   ├── AccMeasure.m                #     Accuracy measurement
│   └── [Other utility functions]   #     Mathematical and evaluation tools
├── demo.m                          # 🎯 Main demonstration entry point
├── examples/                       # 📚 Advanced examples
│   ├── generate_CFSnmiAcc.m        #     Generate CFS performance matrix
│   └── example_CFSnmiAcc.m         #     CFSnmiAcc analysis example
└── DATA_SET/                       # 📁 Data directory
    └── leukemia/                   # 🧬 Leukemia dataset
```

## 🏗️ Modular Architecture

The core algorithm `collaborative_feature_selection.m` has been refactored into 6 independent modules:

### 1. Data Partitioning Module (`partition_data.m`)
- **Function**: Split training data into distributed nodes
- **Input**: Training data, labels, number of nodes
- **Output**: Partitioned data structure

### 2. Intermediate Representation Construction (`construct_intermediate_representation.m`)
- **Function**: Build intermediate representation via kernel locally linear projection
- **Key operations**: KLPP dimensionality reduction, anchor mapping
- **Output**: Intermediate representations for test and anchor points

### 3. Optimal Subspace Construction (`construct_optimal_subspace.m`)
- **Function**: Build joint subspace through SVD
- **Key operations**: Subspace alignment, linear transformation computation
- **Output**: Optimized subspace representation

### 4. Collaborative Optimization Iteration (`collaborative_optimization.m`)
- **Function**: L2,1 regularized iterative optimization
- **Features**: Convergence checking, early stopping, performance optimization
- **Output**: Optimized feature weight matrix

### 5. Feature Ranking Computation (`compute_feature_ranking.m`)
- **Function**: Feature importance ranking based on L2 norm
- **Optional**: Feature importance visualization
- **Output**: Feature ranking indices and importance scores

### 6. Feature Subset Evaluation (`evaluate_feature_subsets.m`)
- **Function**: Incremental feature subset classification performance evaluation
- **Output**: Classification results for different feature subsets

### Modular Advantages
- ✅ **Maintainability**: Each module has single responsibility, easy to understand and modify
- ✅ **Reusability**: Modules can be used independently in other algorithms
- ✅ **Testability**: Each module can be tested and validated independently
- ✅ **Extensibility**: Easy to add new features or optimize specific modules

## 🎯 Use Cases

- **Bioinformatics**: Gene expression data feature selection
- **Privacy protection**: Collaborative learning in distributed environments
- **High-dimensional data**: Dimensionality reduction and selection for large feature sets
- **Federated learning**: Multi-party collaborative data analysis

## 💡 Usage Tips

1. **First run**: Run the main `demo.m` for quick verification
2. **Performance tuning**: Adjust `param.na` and `param.neig` based on data scale
3. **Memory limitations**: If encountering memory issues, reduce anchor count
4. **Result interpretation**: Focus on balance between NMI and accuracy metrics

## 📞 Technical Support

### Common Problem Solutions

**Java AWT Error**
- Use batch mode: `matlab -batch "dataset_choice=1; demo"`
- This is a common issue in command line mode

**Array Dimension Error**
- Array concatenation issues have been fixed
- If similar errors occur, check that label arrays are row vectors

**Slow Execution**
- Adjust `param.na` (anchor count): 50 (fast) → 100 (balanced) → 200 (accurate)
- Reduce `param.neig` (eigenvalue count): 10 → 12 → 18

**General Troubleshooting**
1. Check that data files exist in correct paths
2. Confirm MATLAB version compatibility (recommend R2020a+)
3. Adjust parameters to fit hardware limitations

## 🔬 Algorithm Overview

### Distributed Collaborative Framework

The core algorithm implements a multi-stage distributed learning approach:

1. **Local Feature Learning**: Each data division creates intermediate representations
2. **Anchor-based Alignment**: Uses shared anchor points to align representations across divisions
3. **Collaborative Subspace Construction**: SVD-based optimal subspace construction from all divisions
4. **Iterative Optimization**: Feature selection through iterative matrix optimization with L2,1 regularization
5. **Feature Ranking**: Ranks features by norm of transformation matrix rows

### Key Technical Contributions

- **Privacy-preserving collaboration**: Enables feature selection without sharing raw data
- **Anchor-based alignment**: Novel method for aligning representations across distributed nodes
- **Kernel-based intermediate representation**: Uses KLPP for effective dimensionality reduction
- **Iterative optimization**: Efficient L2,1 regularized optimization with convergence guarantees

---

🌟 **Open source contributions welcome!** If you have improvement suggestions or find bugs, please submit issues or PRs.