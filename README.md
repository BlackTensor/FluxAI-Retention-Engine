# FluxAI — Enterprise Retention Engine ⚡

FluxAI is an end-to-end customer retention and recovery platform designed to identify churn risks with precision and automate the recovery process. By combining state-of-the-art Gradient Boosting (**XGBoost**) with Explainable AI (**SHAP**) and Large Language Models (**Llama 3.2**), FluxAI provides a complete loop from risk detection to personalized retention strategy.

## 📈 Business Impact

FluxAI transforms raw churn data into actionable insights, reducing manual analysis time for Customer Success teams by providing instant, AI-generated recovery strategies. By proactively identifying at-risk customers and providing tailored intervention plans, organizations can significantly improve retention rates and customer lifetime value.

## 🚀 Key Features

- **Predictive Scoring**: Uses a high-performance XGBoost model to score customers based on churn probability.
- **Explainable AI (XAI)**: Integrates SHAP (SHapley Additive exPlanations) to decompose every individual prediction, showing exactly *why* a customer is at risk.
- **Autonomous Recovery Engine**: Leverages local LLMs (Llama 3.2 via Ollama) to generate personalized, empathetic recovery playbooks and emails tailored to each customer's specific risk factors.
- **Interactive Deep Dives**: Drill down into individual customer profiles to view real-time risk drivers and historical tenure data.
- **Enterprise UI**: A premium, responsive Streamlit-based dashboard optimized for customer success teams.

## 🛠️ Tech Stack

- **Frontend/Dashboard**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [Scikit-learn](https://scikit-learn.org/)
- **Model Interpretability**: [SHAP](https://shap.readthedocs.io/)
- **Large Language Model**: [Llama 3.2](https://ollama.com/) (running locally via Ollama)
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly
- **Model Persistence**: Joblib

## 📊 Demo Data

The repository includes a complex test dataset to demonstrate the model’s ability to identify "Ghost Users" and "Angry Veterans" through behavioral signals rather than just contract tenure. This dataset is optimized to show how FluxAI handles various churn personas in a real-world enterprise environment.

## 📦 Project Structure

```text
├── .streamlit/          # Streamlit configuration and theme
├── components/          # Modular UI components (Onboarding, Dashboard, Deep Dive, Recovery)
├── data/               # Dataset storage and generation scripts
├── models/             # Trained model artifacts (.pkl)
├── training/           # Model training and evaluation scripts
├── utils/              # Core logic: ML Predictor, Ollama Client, Data Utilities
└── app.py              # Main application entry point
```

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/BlackTensor/FluxAI-Retention-Engine.git
cd FluxAI-Retention-Engine
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Ollama (Optional - for Recovery Engine)
To use the AI-generated recovery playbooks, install [Ollama](https://ollama.com/) and pull the Llama 3.2 model:
```bash
ollama pull llama3.2
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 📉 Model Training
If you wish to retrain the model on new data:
```bash
python training/train_model.py
```
This script will preprocess the data, train an XGBoost classifier, generate SHAP explainer artifacts, and save everything to the `models/` directory.

## 📄 License
Distributed under the MIT License.

## 👤 Author
**Shayan Ansari**  

---
*Developed as part of a Customer Analytics & AI implementation project.*
