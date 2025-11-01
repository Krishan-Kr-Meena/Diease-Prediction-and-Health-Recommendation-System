from flask import Flask, flash, render_template, redirect, url_for, session, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, EmailField, IntegerField, SelectField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
import bcrypt
from psutil import users
from flask_mysqldb import MySQL
import pickle
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import uuid # Needed for secure token generation
import ast # Added ast import for list parsing

# Suppress minor warnings for a cleaner console output
warnings.filterwarnings('ignore')

# Required imports for loading pickled objects and Keras model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
try:
    # Attempt to import TensorFlow/Keras for the ANN model
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

# --- 1. Global Initialization of ML Artifacts and Data (Unchanged) ---
LABEL_ENCODER = None
SCALER = None
FEATURE_ENGINEER = None
SELECTED_FEATURES = []
SYMPTOMS_DICT = {} 
RAW_DATA = {}
DIAGNOSTIC_SYSTEM = None
models = {} 

TOP_SYMPTOM_NAMES = ['fatigue', 'vomiting', 'high_fever', 'loss_of_appetite', 'nausea', 'headache', 'abdominal_pain', 'yellowish_skin']


# --- 2. FeatureEngineer Class Definition (Unchanged) ---
class FeatureEngineer:
    def __init__(self, symptom_severity_df):
        self.severity_dict = {}
        if isinstance(symptom_severity_df, pd.DataFrame):
             self.severity_dict = dict(zip(
                symptom_severity_df['Symptom'],
                symptom_severity_df['weight']
            ))

    def create_severity_features(self, X):
        X_enhanced = X.copy()
        
        # REINFORCED CHECK: Ensure severity_dict is not empty before iterating
        if not self.severity_dict:
             X_enhanced['total_severity'] = 0
             X_enhanced['avg_severity'] = 0
             X_enhanced['max_severity'] = 0
             X_enhanced['num_symptoms'] = X.sum(axis=1)
             return X_enhanced

        severity_scores = []
        avg_severity = []
        max_severity = []
        for idx, row in X.iterrows():
            present_symptoms = [str(col) for col in X.columns if row[col] == 1] # Ensure column names are strings
            
            total_severity = sum(self.severity_dict.get(col, 0) * row[col] for col in X.columns)
            severity_scores.append(total_severity)
            
            if present_symptoms:
                severities = [self.severity_dict.get(sym, 0) for sym in present_symptoms]
                avg = np.mean(severities) if severities else 0
                max_sev = max(severities) if severities else 0
            else:
                avg = 0
                max_sev = 0
                
            avg_severity.append(avg)
            max_severity.append(max_sev)
            
        X_enhanced['total_severity'] = severity_scores
        X_enhanced['avg_severity'] = avg_severity
        X_enhanced['max_severity'] = max_severity
        X_enhanced['num_symptoms'] = X.sum(axis=1)

        return X_enhanced

    def create_interaction_features(self, X, top_symptoms=10, top_symptom_names=None):
        X_enhanced = X.copy()
        
        if top_symptom_names is None:
            global TOP_SYMPTOM_NAMES
            top_symptom_names = TOP_SYMPTOM_NAMES
            
        if not isinstance(top_symptom_names, list):
            return X_enhanced
            
        for i in range(len(top_symptom_names)):
            for j in range(i + 1, len(top_symptom_names)):
                sym1 = str(top_symptom_names[i])
                sym2 = str(top_symptom_names[j])

                if sym1 in X.columns and sym2 in X.columns:
                    interaction_name = f"{sym1[:15]}_{sym2[:15]}_interact"
                    X_enhanced[interaction_name] = X[sym1] * X[sym2]

        return X_enhanced


# --- 3. ML Model Loading Logic (Unchanged) ---
MODEL_PATH = './models/'
DATA_PATH = './models/medicine-recommendation-system-dataset/' 

try:
    print("--- Loading ML Artifacts ---")
    
    with open(f'{MODEL_PATH}ensemble_models.pkl', 'rb') as f:
        loaded_ensembles = pickle.load(f)
        
    models['Random Forest'] = loaded_ensembles.get('random_forest')
    models['XGBoost'] = loaded_ensembles.get('xgboost')
    models['Voting Ensemble'] = loaded_ensembles.get('voting')
    models['Stacking Ensemble'] = loaded_ensembles.get('stacking') 

    with open(f'{MODEL_PATH}preprocessing_objects.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
        
    with open(f'{MODEL_PATH}symptoms_dict.pkl', 'rb') as f:
        SYMPTOMS_DICT = pickle.load(f)
    
    LABEL_ENCODER = preprocessors.get('label_encoder')
    SCALER = preprocessors.get('scaler')
    SELECTED_FEATURES = preprocessors.get('selected_features')
    
    print("✅ Ensembles, Preprocessors, and Symptom Dict loaded.")

except FileNotFoundError as e:
    print(f"❌ FATAL: A critical model file was not found. Check paths in '{MODEL_PATH}'. Error: {e}")
except Exception as e:
    print(f"❌ FATAL: Failed to load core ML artifacts: {e}")

try:
    if load_model and 'Stacking Ensemble' in models:
        models['Neural Network'] = load_model(f'{MODEL_PATH}neural_network_model.h5')
        print("✅ Neural Network model loaded.")
    elif not load_model:
        print("⚠️ TensorFlow not imported. Neural Network model not loaded.")
        
except Exception as e:
    print(f"❌ Failed to load Neural Network model: {e}")

try:
    print("--- Loading Raw Data ---")
    RAW_DATA['description'] = pd.read_csv(f'{DATA_PATH}description.csv')
    RAW_DATA['medications'] = pd.read_csv(f'{DATA_PATH}medications.csv')
    RAW_DATA['precautions'] = pd.read_csv(f'{DATA_PATH}precautions_df.csv')
    RAW_DATA['symptom_severity'] = pd.read_csv(f'{DATA_PATH}Symptom-severity.csv')
    RAW_DATA['diets'] = pd.read_csv(f'{DATA_PATH}diets.csv')
    RAW_DATA['workout'] = pd.read_csv(f'{DATA_PATH}workout_df.csv')
    
    FEATURE_ENGINEER = FeatureEngineer(RAW_DATA['symptom_severity'])

    print("✅ Raw Recommendation Data loaded.")
    
except Exception as e:
    print(f"⚠️ Failed to load raw data for recommendations. Check paths in '{DATA_PATH}'. Error: {e}")


# --- 4. Define Prediction Logic Class (Modified get_medical_info) ---
class MedicalDiagnosticSystem:
    def __init__(self, models_dict, label_encoder, scaler, feature_engineer, selected_features, top_symptom_names, symptoms_dict, data):
        self.models = models_dict
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.feature_engineer = feature_engineer
        self.selected_features = selected_features
        self.top_symptom_names = top_symptom_names
        self.symptoms_dict = symptoms_dict
        self.data = data
        self.X_original_cols = [str(k) for k in symptoms_dict.keys()]

    def preprocess_symptoms(self, symptoms):
        valid_symptoms = []
        invalid_symptoms = []
        for symptom in symptoms:
            symptom_clean = str(symptom).lower().strip().replace(' ', '_')
            if symptom_clean in self.symptoms_dict:
                valid_symptoms.append(symptom_clean)
            else:
                invalid_symptoms.append(symptom)
        return valid_symptoms, invalid_symptoms

    def create_features(self, symptoms):
        base_vector = np.zeros(len(self.symptoms_dict))
        symptom_to_index = {str(k): v for k, v in self.symptoms_dict.items()}
        
        for symptom in symptoms:
            symptom_key = str(symptom)
            if symptom_key in symptom_to_index:
                 base_vector[symptom_to_index[symptom_key]] = 1

        base_df = pd.DataFrame([base_vector], columns=self.X_original_cols)
        
        enhanced_df = self.feature_engineer.create_severity_features(base_df)
        enhanced_df = self.feature_engineer.create_interaction_features(
            enhanced_df, top_symptom_names=self.top_symptom_names
        )

        final_features = pd.DataFrame(0, index=enhanced_df.index, columns=self.selected_features)
        
        for feature in self.selected_features:
            if feature in enhanced_df.columns:
                final_features[feature] = enhanced_df[feature]
            else:
                final_features[feature] = 0 
        return final_features.values[0]

    # --- START MODIFIED get_medical_info FUNCTION ---
    def get_medical_info(self, disease):
        info = {'description': 'No description available.', 'precautions': [], 'medications': [], 'diet': [], 'workout': []}
        disease_clean = str(disease).strip()
        
        DEFAULT_WORKOUT = "Follow a balanced and nutritious diet"

        # Helper function for robust lookup and list parsing (used for medications, diets)
        def safe_lookup_and_parse(df_key, disease_col, data_col):
            df = self.data.get(df_key)
            if df is None:
                return []

            match_col = disease_col if disease_col in df.columns else None

            if not match_col or data_col not in df.columns:
                return []

            match = df[df[match_col].astype(str).str.strip() == disease_clean][data_col]
            
            if match.empty:
                return []
            
            value = match.values[0]
            
            if isinstance(value, list):
                return [str(v) for v in value if pd.notna(v)]

            value_string = str(value).strip()
            if value_string.startswith("[") and value_string.endswith("]"):
                try:
                    parsed_list = ast.literal_eval(value_string)
                    if isinstance(parsed_list, list):
                        return [str(v).strip().strip("'") for v in parsed_list if pd.notna(v)]
                except (ValueError, SyntaxError):
                    pass

            if pd.notna(value) and value_string:
                return [value_string]
                
            return []

        # --- DEDICATED WORKOUT LOOKUP (FIXED) ---
        def get_workout_list(disease_clean):
            df = self.data.get('workout')
            if df is None or 'disease' not in df.columns or 'Lifestyle & workout' not in df.columns:
                return [DEFAULT_WORKOUT] 
            
            matches = df[df['disease'].astype(str).str.strip() == disease_clean]['Lifestyle & workout']
            
            if matches.empty:
                return [DEFAULT_WORKOUT]
            
            workout_items = [str(w).strip() for w in matches.values if pd.notna(w)]
            workout_items = [item for item in workout_items if item]
            
            if not workout_items:
                 return [DEFAULT_WORKOUT]

            return workout_items
        # --- END DEDICATED WORKOUT LOOKUP ---


        # Description
        desc_df = self.data.get('description')
        desc_match = desc_df[desc_df['Disease'].astype(str).str.strip() == disease_clean]['Description']
        if not desc_match.empty:
            info['description'] = str(desc_match.values[0])

        # --- START FIX: Robust Precaution Lookup ---
        prec_df = self.data.get('precautions')
        if prec_df is not None and 'Disease' in prec_df.columns:
            prec_match = prec_df[prec_df['Disease'].astype(str).str.strip() == disease_clean]
            
            if not prec_match.empty:
                prec_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
                
                # Extract values from all precaution columns for the matched row
                prec_values = prec_match[prec_cols].values[0] 
                
                # FIX: Use a more robust check to filter out NaN/None/Empty strings
                info['precautions'] = [
                    str(p).strip() 
                    for p in prec_values 
                    if pd.notna(p) and str(p).strip() != '' 
                ]
        # --- END FIX: Robust Precaution Lookup ---
        
        # Medications
        info['medications'] = safe_lookup_and_parse('medications', 'Disease', 'Medication')
        
        # DIETS 
        info['diet'] = safe_lookup_and_parse('diets', 'Disease', 'Diet')
        
        # Workout
        info['workout'] = get_workout_list(disease_clean)

        return info
    # --- END MODIFIED get_medical_info FUNCTION ---

    def predict(self, symptoms, model_name='Stacking Ensemble', top_n=3):
        if not self.label_encoder or not self.scaler or not self.feature_engineer:
            return {'error': 'ML system is not fully initialized.', 'success': False, 'invalid_symptoms': symptoms}

        valid_symptoms, invalid_symptoms = self.preprocess_symptoms(symptoms)

        if not valid_symptoms:
            return {'error': 'No valid symptoms provided', 'success': False, 'invalid_symptoms': invalid_symptoms}

        feature_vector = self.create_features(valid_symptoms)

        if model_name not in self.models or self.models[model_name] is None:
            return {'error': f'Model {model_name} not found in loaded artifacts.', 'success': False, 'invalid_symptoms': invalid_symptoms}

        model = self.models[model_name]

        if model_name == 'Neural Network':
            feature_vector_scaled = self.scaler.transform([feature_vector])
            probabilities = model.predict(feature_vector_scaled, verbose=0)[0]
        else:
            probabilities = model.predict_proba([feature_vector])[0]

        top_n_idx = np.argsort(probabilities)[::-1][:top_n]

        predictions = []
        for idx in top_n_idx:
            disease = str(self.label_encoder.classes_[idx]) 
            confidence = probabilities[idx]

            predictions.append({
                'disease': disease,
                'confidence': float(confidence),
                'medical_info': self.get_medical_info(disease)
            })

        return {
            'success': True,
            'model_used': model_name,
            'symptoms_analyzed': valid_symptoms,
            'invalid_symptoms': invalid_symptoms,
            'predictions': predictions,
            'primary_diagnosis': predictions[0]['disease'],
            'primary_confidence': predictions[0]['confidence'],
        }

# --- 5. Initialize the Diagnostic System (Unchanged) ---
if models.get('Stacking Ensemble') and LABEL_ENCODER and FEATURE_ENGINEER:
    DIAGNOSTIC_SYSTEM = MedicalDiagnosticSystem(
        models_dict=models,
        label_encoder=LABEL_ENCODER,
        scaler=SCALER,
        feature_engineer=FEATURE_ENGINEER,
        selected_features=SELECTED_FEATURES,
        top_symptom_names=TOP_SYMPTOM_NAMES,
        symptoms_dict=SYMPTOMS_DICT,
        data=RAW_DATA
    )
    print("✅ Medical Diagnostic System Initialized.")
else:
    print("❌ Critical models/preprocessors are missing. Prediction service is disabled.")


# --- 6. Flask App Setup (Unchanged) ---
app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'user_registration'
app.config['SECRET_KEY'] = 'your_secret_key'
mysql = MySQL(app)

# --- 7. Flask Forms (Unchanged) ---
class RegistrationForm(FlaskForm):
    first_name = StringField('First Name', render_kw={"placeholder": "Enter your first name"}, validators=[DataRequired()])
    last_name = StringField('Last Name', render_kw={"placeholder": "Enter your last name"}, validators=[DataRequired()])
    age = IntegerField('Age', render_kw={"placeholder": "Enter your age"}, validators=[DataRequired()])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')], validators=[DataRequired()])
    email = EmailField('Email', render_kw={"placeholder": "Enter your email"}, validators=[DataRequired(), Email()])
    password = PasswordField('Password', render_kw={"placeholder": "Enter your password"}, validators=[DataRequired(), Length(min=8)])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Register')

class SigninForm(FlaskForm):
    email = EmailField('Email', render_kw={"placeholder": "Enter your email"}, validators=[DataRequired(), Email()])
    password = PasswordField('Password', render_kw={"placeholder": "Enter your password"}, validators=[DataRequired(), Length(min=8)])
    submit = SubmitField('Sign In')

class RequestResetForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', 
                             validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm New Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')


# --- 8. Helper Functions for Password Reset (Unchanged) ---
reset_tokens = {} 

def get_user_by_email(email):
    """Fetches user from DB by email."""
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM users WHERE email = %s", [email])
    user = cursor.fetchone() 
    cursor.close()
    return user

def update_user_password(email, new_password):
    """Hashes and updates the user's password in the database."""
    try:
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        
        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE users SET password = %s WHERE email = %s", 
                       (hashed_password, email))
        mysql.connection.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"Database update failed: {e}")
        return False

def generate_reset_token(email):
    """Generates a unique, one-time use token."""
    token = str(uuid.uuid4())
    reset_tokens[token] = email 
    return token

def verify_reset_token(token, consume=False):
    """
    Verifies the token.
    If consume=True, it removes the token from the store (marks as used).
    """
    email = reset_tokens.get(token)
    if email and consume:
        del reset_tokens[token] 
    return email


# --- 9. Flask Routes (Unchanged) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    form = SigninForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id, password FROM users WHERE email=%s", [email])
        user_data = cursor.fetchone()
        cursor.close()
        
        if user_data:
            user_id, hashed_password = user_data
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                session['user_id'] = user_id
                flash('Login Successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Login Unsuccessful. Please check email and password', 'danger')
        else:
             flash('Login Unsuccessful. Please check email and password', 'danger')
            
    return render_template('signin.html', form=form) 


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        first_name = form.first_name.data
        last_name = form.last_name.data
        age = form.age.data
        gender = form.gender.data
        email = form.email.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id FROM users WHERE email = %s", [email])
        if cursor.fetchone():
            flash('An account with this email already exists.', 'warning')
            cursor.close()
            return redirect(url_for('register'))

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (first_name, last_name, age, gender, email, password) VALUES (%s, %s, %s, %s, %s, %s)",
                       (first_name, last_name, age, gender, email, hashed_password))
        mysql.connection.commit()
        cursor.close()
        
        flash('Registration successful! Please sign in.', 'success')
        return redirect(url_for('signin'))

    return render_template('register.html', form=form)

@app.route("/forgot_password", methods=['GET', 'POST'])
def forgot_password():
    if 'user_id' in session:
        return redirect(url_for('dashboard')) 
        
    form = RequestResetForm()
    
    if form.validate_on_submit():
        user_email = form.email.data
        user = get_user_by_email(user_email)
        
        if user:
            token = generate_reset_token(user_email) 
            flash(f'An email with password reset instructions has been sent to {user_email}.', 'info')
            
            return redirect(url_for('reset_password', token=token))
        else:
            flash('If an account with that email exists, an email with reset instructions has been sent.', 'info')

            
    return render_template('forgot_password.html', title='Forgot Password', form=form)


@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_password(token):
    email = verify_reset_token(token, consume=False) 

    if email is None:
        flash('That is an invalid or expired reset link. Please request a new one.', 'danger')
        return redirect(url_for('forgot_password'))

    form = ResetPasswordForm()
    
    if form.validate_on_submit():
        new_password = form.password.data
        
        if update_user_password(email, new_password):
            verify_reset_token(token, consume=True) 
            flash('Your password has been updated successfully! Please sign in.', 'success')
            return redirect(url_for('signin')) 
        else:
            flash('Could not update password due to a system error. Please try again or contact support.', 'danger')
            
    return render_template('reset_password.html', title='Reset Password', form=form, token=token)


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('You need to log in to access the dashboard.', 'info')
        return redirect(url_for('signin'))

    user_id = session['user_id']
    
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT first_name, last_name, email, age, gender FROM users WHERE id = %s", [user_id])
    user_info = cursor.fetchone()
    cursor.close()

    if user_info:
        return render_template('dashboard.html', user_data=user_info)
    else:
        flash('User data not found.', 'danger')
        return redirect(url_for('signin'))

@app.route('/logout')
def logout():
    if '_flashes' in session:
        session.pop('_flashes', None)

    session.pop('user_id', None)
    
    flash('You have been logged out successfully.', 'success') 
    
    return redirect(url_for('signin'))

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/med_service')
def med_service():
    all_symptoms = list(SYMPTOMS_DICT.keys()) 
    return render_template('med_service.html', all_symptoms=all_symptoms)

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if DIAGNOSTIC_SYSTEM is None:
        flash("System Error: The medical diagnostic model failed to load during startup. Please check the console logs for details.", 'danger')
        return redirect(url_for('med_service'))
        
    if 'user_id' not in session:
        flash('Please log in to use the diagnostic service.', 'info')
        return redirect(url_for('signin'))

    symptom_input = request.form.get('symptoms')
    model_choice = request.form.get('model_choice', '1')
    
    if not symptom_input:
        flash("Please enter at least one symptom.", 'warning')
        return redirect(url_for('med_service'))
    
    input_symptoms = [s.strip() for s in symptom_input.split(',') if s.strip()]
    
    model_map = {
        '1': 'Stacking Ensemble',
        '2': 'Neural Network',
        '3': 'Random Forest'
    }
    model_name = model_map.get(model_choice, 'Stacking Ensemble')

    try:
        prediction_result = DIAGNOSTIC_SYSTEM.predict(
            symptoms=input_symptoms, 
            model_name=model_name
        )

        if prediction_result['success']:
            warning = None
            if prediction_result['primary_confidence'] < 0.70:
                 warning = f"Low confidence ({prediction_result['primary_confidence']*100:.1f}%). Professional consultation strongly recommended."
            
            return render_template('med_service_result.html', 
                                   result=prediction_result,
                                   input_symptoms=symptom_input,
                                   warning=warning)
        else:
            flash(f"Prediction Error: {prediction_result.get('error', 'Unknown error.')}", 'danger')
            return redirect(url_for('med_service'))

    except Exception as e:
        error_message = str(e)
        flash(f"An unexpected system error occurred: {error_message}", 'danger')
        print(f"Prediction Error Trace: {error_message}")
        return redirect(url_for('med_service'))


@app.route('/doc_service')
def doc_service():
    return render_template('doc_services.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

if __name__ == '__main__':
    app.run(debug=True)
