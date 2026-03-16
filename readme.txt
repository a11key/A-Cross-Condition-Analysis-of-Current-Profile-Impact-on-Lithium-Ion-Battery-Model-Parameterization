## Project Structure

### Data_processing/
Contains the data, processing scripts, and results obtained during the analysis.

#### Data_preprocessing/
Preprocessing of raw data:
- parsing raw datasets
- creating separate files with battery profiles
- extracting charge and discharge curves

#### Parameters/
Parameterization of the battery model:
- **Static parameters**: `Q`, `η`, `OCV(SOC)`
- **Dynamic parameters**: `R₀`, `Rᵢ`, `τᵢ`

#### Results/
Validation and post-processing of validation results.
Includes analysis, visualization, and evaluation of model performance.
