The Medical Appointment No-Show Predictor predicts a patient's likelihood of missing a scheduled appointment.
The system deploys a Suport Vector Machine with an RBF kernel to make its risk output. It posses as a tool reciptionists of hospitals use to identify which patient requires extra absenteeism avoidance strategies to be used on.
The program takes in patient data such as, Age, Gender, Days until Appointment, SMS reinder status, Hypertension status, Diabetes status, Alcoholism status and Handicap level and outputs a risk score iin percentage as well as a binary prediction.

The dataset used in training the model was Kaggle's "Medical Appointment No Shows" which contained approximately 110,000 records. 
The project structure is as follows:
NoShowPredictor/
├── app.py
├── svm_model.pkl
├── svm_scaler.pkl
├── svm_feature_columns.pkl
├── README.md
├── templates/
│   └── index.html
└── training/
    ├── svm_final_project.ipynb       # Optional training notebook
    └── KaggleV2-May-2016.csv   

The .pkl files are saved files from the app that separate the program into the model, scaler and features for app.py to use.
The notebook and dataset are only required for training or reproducing the model. They are not required to run the web application after the `.pkl` files have been created.

**How To Run The Program**
1. Once the project structure has been followed, open the folder in your IDE.
2. Create a virtual environment and install all dependencies and libraries in requirements.txt
3. Run python app.py or python3 app.py
4. http://127.0.0.1:5000 will appear. Copy and paste it into the url bar in your browser and hit enter.
