## Model Card for UniFIDES: Universal Fractional Integro-Differential Equation Solvers

### Model Details
- **Developer**: Milad Saadat, Deepak Mangal, Safa Jamali
- **Release Date**: April 27, 2024
- **Model Version**: 1.0
- **License**: MIT
- **Model Type**: TensorFlow implementation
- **Publication**: Submitted to Nature Machine Intelligence

### Intended Use
- **Primary Use**: To solve integer-order and fractional fractional integro-differential equations (FIDEs) in forward and inverse directions.
- **Intended Users**: Researchers and professionals in fields such as physics, engineering, and applied mathematics who need to solve complex differential equations.
- **Out-of-Scope Uses**: The tool is not intended for non-scientific computations or for solving equations beyond the specified types of integro-differential equations.

### Model/Data Inputs and Outputs
- **Inputs**: Mathematical expressions of integro-differential equations, including but not limited to Fredholm and Volterra types.
- **Outputs**: Solutions to the input equations, presented as visual plots and data outputs within a Jupyter Notebook environment.

### Model Performance
- **Metrics**: Accuracy of the solutions compared to known solutions (where available) and computational efficiency.
- **Results**: Demonstrated to accurately solve a variety of integro-differential equations efficiently, including test cases of integer and fractional orders.

### Ethical Considerations
- Care should be taken when using this model in critical applications as incorrect setup or bugs may lead to incorrect solutions that could influence subsequent decision-making.

### Caveats and Recommendations
- Users should ensure they understand the mathematical and computational complexities involved in their specific use case.
- Ensure that the TensorFlow and other dependency versions are compatible with the users’ systems to avoid computational discrepancies.

### Contributors
- **Milad Saadat**: [Google Scholar Profile](https://scholar.google.com/citations?user=PPLvVmEAAAAJ&hl=en&authuser=1)
- **Deepak Mangal**: [Google Scholar Profile](https://scholar.google.com/citations?user=AoYKLW4AAAAJ&hl=en)
- **Safa Jamali**: [Google Scholar Profile](https://scholar.google.com/citations?user=D1asaYIAAAAJ&hl=en)
- **Acknowledgements**: Discussions with Dr. Deepak Mangal and support from the National Science Foundation’s DMREF program through Award \#2118962.

