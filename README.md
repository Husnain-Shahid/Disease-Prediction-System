# 🏥 Disease Prediction System

A machine learning-powered web application that predicts diseases based on patient symptoms. The system uses an ensemble of advanced classifiers and a comprehensive symptom questionnaire organized across 10 medical categories.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![SQLite](https://img.shields.io/badge/SQLite-3.0%2B-lightblue)

---

## 🎯 Features

- **🔍 Intelligent Disease Prediction**: Uses ensemble machine learning models to accurately predict diseases
- **📋 Comprehensive Symptom Assessment**: 340+ yes/no questions across 10 medical categories
- **💾 Patient History Tracking**: Stores and retrieves patient medical history with predictions
- **📊 Confidence Scoring**: Returns prediction confidence for medical decision support
- **💊 Medicine Recommendations**: Suggests medicines based on predicted disease
- **🏥 Medical Advice**: Provides guidance and precautions for each disease
- **🌐 Web Interface**: User-friendly Flask-based web application
- **📱 Responsive Design**: Clean and intuitive UI with CSS styling

---

## 🏗️ Project Structure

```
Disease-Prediction-System/
├── app.py                          # Flask application & routes
├── model.py                        # ML model training & pipeline
├── database.py                     # SQLite database management
├── symptoms_questionnaire.py       # Symptom questions & categories
├── generate_dataset.py             # Dataset generation utilities
├── migrate_database.py             # Database migration script
│
├── saved_model/                    # Pre-trained model files
│   ├── pipeline.pkl                # Scikit-learn pipeline
│   ├── feature_columns.pkl         # Feature names
│   ├── model.pkl                   # Trained classifier
│   ├── encoders.pkl                # Label encoders
│   └── target_encoder.pkl          # Disease target encoder
│
├── templates/                      # HTML templates
│   ├── index.html                  # Main prediction interface
│   └── history.html                # Patient history page
│
├── static/                         # Static assets
│   ├── style.css                   # Styling
│   └── script.js                   # Frontend JavaScript
│
├── datasets/                       # Training datasets
│   ├── Disease and symptoms dataset.csv
│   ├── Disease_dataset_10000.csv
│   └── Disease_symptom_and_patient_profile_dataset.csv
│
├── SYMPTOM_QUESTIONNAIRE_GUIDE.md # Detailed questionnaire documentation
├── README.md                       # This file
└── .env                            # Environment variables (create as needed)
```

---

## 🔧 Technologies Used

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, Flask |
| **ML Framework** | Scikit-learn |
| **Database** | SQLite3 |
| **Frontend** | HTML5, CSS3, JavaScript |
| **ML Models** | RandomForest, ExtraTreesClassifier, HistGradientBoosting, VotingClassifier |

---

## 📊 Medical Categories

The system assesses symptoms across 10 medical specialties:

1. **Respiratory & ENT** (30 questions) - Breathing, throat, nose, ear
2. **Cardiac & Circulatory** (14 questions) - Heart, chest, blood pressure
3. **Gastrointestinal** (22 questions) - Stomach, digestive, bowel
4. **Urinary** (14 questions) - Kidney, bladder, urinary function
5. **Reproductive & Breast** (41 questions) - Sexual and reproductive health
6. **Neurological** (32 questions) - Brain, nervous system, mental health
7. **Musculoskeletal** (77 questions) - Joints, muscles, bones
8. **Skin & Hair** (30 questions) - Skin conditions, rash, hair
9. **Eye & Vision** (22 questions) - Eye health, vision
10. **General & Systemic** (26 questions) - Whole-body symptoms

**Total: 340+ questions**

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- Git

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Disease-Prediction-System
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Initialize Database
```bash
python migrate_database.py
```

### Step 5: Train Model (Optional - Pre-trained model included)
```bash
python model.py
```

### Step 6: Run Application
```bash
python app.py
```

Visit `http://localhost:5000` in your browser

---

## 💻 Usage

### For End Users
1. **Open the web application** at `http://localhost:5000`
2. **Fill in patient details** (name, age, gender)
3. **Answer symptom questionnaire** - Select "Yes" or "No" for each symptom
4. **Get prediction** - View predicted disease with confidence score
5. **View recommendations** - See suggested medicines and medical advice
6. **Check history** - Navigate to history page to view past predictions

### For Developers

#### Making Predictions Programmatically
```python
import pickle
import pandas as pd

# Load model
model = pickle.load(open("saved_model/pipeline.pkl", "rb"))
feature_columns = pickle.load(open("saved_model/feature_columns.pkl", "rb"))

# Prepare symptom data
symptoms_dict = {
    'shortness_of_breath': 1,
    'persistent_cough': 1,
    'fever': 1,
    # ... more symptoms
}

# Create DataFrame with feature columns
X = pd.DataFrame([symptoms_dict], columns=feature_columns)

# Predict
prediction = model.predict(X)
confidence = model.predict_proba(X).max()
```

#### Adding New Symptoms
Edit `symptoms_questionnaire.py`:
```python
SYMPTOM_QUESTIONNAIRE = {
    "Category_Name": [
        "Your new question?",
        # ... other questions
    ]
}
```

---

## 📈 Machine Learning Model

### Architecture
- **Ensemble Voting Classifier** combining:
  - Random Forest (100 trees)
  - Extra Trees (100 trees)
  - HistGradient Boosting
  
### Training Pipeline
```
Raw Data → Data Cleaning → Normalization → Feature Selection → 
Encoding → Train-Test Split → Model Training → Validation
```

### Model Performance
- **Training Data**: Disease & symptoms dataset (10,000+ samples)
- **Minimum Samples per Disease**: 10 (rare disease filtering)
- **Cross-Validation**: Stratified K-Fold
- **Output**: Disease prediction + confidence score (0-1)

### Feature Engineering
- Binary normalization of symptom values
- Variance threshold feature selection
- Label encoding of categorical disease names
- Pipeline serialization for deployment

---

## 💾 Database Schema

### History Table
```sql
CREATE TABLE history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT NOT NULL,
    patient_age INTEGER NOT NULL,
    patient_gender TEXT NOT NULL,
    disease TEXT NOT NULL,
    confidence REAL,
    medicines TEXT,
    advice TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔐 Security Notes

- **Data Privacy**: Patient records stored locally in SQLite
- **HIPAA Compliance**: Implement additional measures before production use
- **Input Validation**: All user inputs are validated and sanitized
- **Model Security**: Serialized models should be protected with access controls
- **Recommendations**: 
  - Use HTTPS in production
  - Implement user authentication
  - Add audit logging
  - Regular security updates

---

## 📝 Dataset Information

### Included Datasets
1. **Disease and symptoms dataset.csv** - Main training data
2. **Disease_dataset_10000.csv** - Extended dataset
3. **Disease_symptom_and_patient_profile_dataset.csv** - Comprehensive patient profiles

### Data Characteristics
- **Diseases**: Multiple chronic and acute diseases
- **Symptoms**: Binary (yes/no) and continuous values
- **Patient Info**: Age, gender, medical history
- **Total Samples**: 10,000+

### Generating Custom Datasets
```bash
python generate_dataset.py
```

---

## 🛠️ Configuration

Create a `.env` file in the project root:
```env
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///history.db
```

---

## 📚 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main prediction page |
| GET | `/history` | Patient history page |
| POST | `/predict` | Make disease prediction |
| GET | `/get_history` | Fetch all patient records |
| GET | `/download_history` | Download history as PDF |

---

## 🐛 Troubleshooting

### Model Not Loading
```bash
# Retrain model
python model.py
```

### Database Errors
```bash
# Reset database
rm history.db
python migrate_database.py
```

### Missing Dependencies
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### Port Already in Use
```bash
# Use different port
python app.py --port 5001
```

---

## 🔮 Future Improvements

- [ ] Add user authentication & role-based access
- [ ] Implement advanced analytics dashboard
- [ ] Export predictions to PDF/Excel
- [ ] Integration with medical databases
- [ ] Mobile app development
- [ ] Real-time model updates
- [ ] Explainable AI (SHAP) integration
- [ ] Multi-language support
- [ ] Telemedicine integration
- [ ] API for external applications

---

## 📖 Documentation

- [Symptom Questionnaire Guide](SYMPTOM_QUESTIONNAIRE_GUIDE.md) - Detailed questionnaire structure
- [Model Training Details](model.py) - Machine learning implementation

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

**This system is for educational and research purposes only.** It should NOT be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

Disease Prediction System Team

---

## 📞 Support

For issues, questions, or suggestions, please:
- Open an issue on GitHub
- Contact the development team
- Check existing documentation

---

## 🙏 Acknowledgments

- Scikit-learn team for excellent ML library
- Flask community for web framework
- Medical datasets contributors
- Open-source community

---

**Last Updated**: 2026-06-23  
**Version**: 1.0.0
