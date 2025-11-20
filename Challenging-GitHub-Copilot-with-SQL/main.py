import sqlite3
from datetime import datetime, timedelta
import random

def get_patients_by_dob_range(start_date, end_date):
    """
    Generate SQL query to get all patients information based on date of birth range.
    
    Args:
        start_date: Start date of birth range (YYYY-MM-DD format)
        end_date: End date of birth range (YYYY-MM-DD format)
    
    Returns:
        str: SQL query string
    """
    query = """
    SELECT *
    FROM patients
    WHERE date_of_birth BETWEEN ? AND ?
    ORDER BY date_of_birth;
    """
    return query


# Example usage:
# sql_query = get_patients_by_dob_range('1980-01-01', '1990-12-31')
# Create sample database with patients
def create_sample_database():
    """Create a sample database with patient information."""
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    
    # Create patients table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        date_of_birth DATE NOT NULL,
        gender TEXT NOT NULL,
        race TEXT NOT NULL,
        smoke_history TEXT,
        drinking_history TEXT
    )
    ''')
    
    # Create medications table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patient_medications (
        medication_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        medication_name TEXT NOT NULL,
        dosage TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    )
    ''')
    
    # Create diseases table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patient_diseases (
        disease_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        disease_name TEXT NOT NULL,
        diagnosis_date DATE,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    )
    ''')
    
    # Sample data
    first_names = ['John', 'Mary', 'James', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    genders = ['Male', 'Female']
    races = ['White', 'Black', 'Asian', 'Hispanic', 'Native American', 'Pacific Islander']
    smoke_histories = ['Never', 'Former', 'Current', 'Occasional']
    drinking_histories = ['Never', 'Occasional', 'Moderate', 'Heavy', 'Former']
    medications = ['Lisinopril', 'Metformin', 'Atorvastatin', 'Amlodipine', 'Omeprazole', 'Levothyroxine', 'Albuterol']
    diseases = ['Hypertension', 'Diabetes Type 2', 'Asthma', 'COPD', 'Depression', 'Arthritis', 'Hypothyroidism']
    
    # Insert sample patients
    for i in range(50):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        # Generate random date of birth between 1940 and 2005
        days_old = random.randint(365*19, 365*84)
        date_of_birth = (datetime.now() - timedelta(days=days_old)).strftime('%Y-%m-%d')
        gender = random.choice(genders)
        race = random.choice(races)
        smoke_history = random.choice(smoke_histories)
        drinking_history = random.choice(drinking_histories)
        
        cursor.execute('''
        INSERT INTO patients (first_name, last_name, date_of_birth, gender, race, smoke_history, drinking_history)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (first_name, last_name, date_of_birth, gender, race, smoke_history, drinking_history))
        
        patient_id = cursor.lastrowid
        
        # Add 0-3 medications per patient
        num_medications = random.randint(0, 3)
        for _ in range(num_medications):
            medication = random.choice(medications)
            dosage = f"{random.choice([5, 10, 20, 50, 100])}mg"
            cursor.execute('''
            INSERT INTO patient_medications (patient_id, medication_name, dosage)
            VALUES (?, ?, ?)
            ''', (patient_id, medication, dosage))
        
        # Add 0-2 diseases per patient
        num_diseases = random.randint(0, 2)
        for _ in range(num_diseases):
            disease = random.choice(diseases)
            diagnosis_date = (datetime.now() - timedelta(days=random.randint(30, 3650))).strftime('%Y-%m-%d')
            cursor.execute('''
            INSERT INTO patient_diseases (patient_id, disease_name, diagnosis_date)
            VALUES (?, ?, ?)
            ''', (patient_id, disease, diagnosis_date))
    
    conn.commit()
    conn.close()
    print("Sample database created successfully with 50 patients!")

# Create the database
create_sample_database()

# Example usage with the existing function:
sql_query = get_patients_by_dob_range('1980-01-01', '1990-12-31')
conn = sqlite3.connect('patients.db')
cursor = conn.cursor()
cursor.execute(sql_query, ('1980-01-01', '1990-12-31'))
results = cursor.fetchall()
print(f"\nFound {len(results)} patients born between 1980-01-01 and 1990-12-31")
def get_patients_by_criteria(gender=None, min_age=None, max_age=None, smoke_history=None, drinking_history=None):
    """
    Generate SQL query to get patients based on multiple criteria.
    
    Args:
        gender: Gender filter (e.g., 'Male', 'Female')
        min_age: Minimum age in years
        max_age: Maximum age in years
        smoke_history: Smoking history filter (e.g., 'Never', 'Former', 'Current', 'Occasional')
        drinking_history: Drinking history filter (e.g., 'Never', 'Occasional', 'Moderate', 'Heavy', 'Former')
    
    Returns:
        tuple: (SQL query string, list of parameters)
    """
    query = "SELECT * FROM patients WHERE 1=1"
    params = []
    
    if gender:
        query += " AND gender = ?"
        params.append(gender)
    
    if min_age is not None:
        max_dob = (datetime.now() - timedelta(days=min_age*365)).strftime('%Y-%m-%d')
        query += " AND date_of_birth <= ?"
        params.append(max_dob)
    
    if max_age is not None:
        min_dob = (datetime.now() - timedelta(days=max_age*365)).strftime('%Y-%m-%d')
        query += " AND date_of_birth >= ?"
        params.append(min_dob)
    
    if smoke_history:
        query += " AND smoke_history = ?"
        params.append(smoke_history)
    
    if drinking_history:
        query += " AND drinking_history = ?"
        params.append(drinking_history)
    
    query += " ORDER BY last_name, first_name;"
    
    return query, params


# Example usage with the new function:
print("\nExample: Female patients, age 30-50, who never smoked, occasional drinkers")
sql_query, params = get_patients_by_criteria(
    gender='Female',
    min_age=30,
    max_age=50,
    smoke_history='Never',
    drinking_history='Occasional'
)
cursor.execute(sql_query, params)
results = cursor.fetchall()
print(f"Found {len(results)} patients matching criteria")
for row in results[:5]:  # Show first 5 results
    print(f"  {row[1]} {row[2]}, DOB: {row[3]}, Gender: {row[4]}")

conn.close()

def create_medications_reference_table():
    """Create a reference table for medications with their usage and side effects."""
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    
    # Create medications reference table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS medications_reference (
        medication_id INTEGER PRIMARY KEY AUTOINCREMENT,
        medication_name TEXT NOT NULL UNIQUE,
        common_dosages TEXT NOT NULL,
        usage TEXT NOT NULL,
        side_effects TEXT NOT NULL
    )
    ''')
    
    # Sample medication data with usage and side effects
    medications_data = [
        ('Lisinopril', '5mg, 10mg, 20mg, 40mg', 
         'Treatment of high blood pressure and heart failure',
         'Dizziness, headache, persistent cough, fatigue'),
        ('Metformin', '500mg, 850mg, 1000mg',
         'Treatment of type 2 diabetes by controlling blood sugar levels',
         'Nausea, diarrhea, stomach upset, metallic taste'),
        ('Atorvastatin', '10mg, 20mg, 40mg, 80mg',
         'Lowering cholesterol and reducing risk of heart disease',
         'Muscle pain, digestive problems, increased blood sugar'),
        ('Amlodipine', '2.5mg, 5mg, 10mg',
         'Treatment of high blood pressure and chest pain (angina)',
         'Swelling of ankles/feet, dizziness, flushing, fatigue'),
        ('Omeprazole', '10mg, 20mg, 40mg',
         'Treatment of acid reflux, heartburn, and stomach ulcers',
         'Headache, stomach pain, nausea, diarrhea, gas'),
        ('Levothyroxine', '25mcg, 50mcg, 75mcg, 100mcg, 125mcg',
         'Treatment of hypothyroidism (underactive thyroid)',
         'Hair loss, weight changes, headache, insomnia'),
        ('Albuterol', '90mcg (inhaler), 2mg, 4mg (tablets)',
         'Relief of bronchospasm in asthma and COPD',
         'Tremors, nervousness, headache, rapid heartbeat')
    ]
    
    # Insert medication reference data
    for med_data in medications_data:
        cursor.execute('''
        INSERT OR IGNORE INTO medications_reference (medication_name, common_dosages, usage, side_effects)
        VALUES (?, ?, ?, ?)
        ''', med_data)
    
    conn.commit()
    conn.close()
    print("\nMedications reference table created successfully!")

# Create the medications reference table
create_medications_reference_table()

# Display the medications reference table
print("\nMedications Reference Table:")
print("-" * 100)
conn = sqlite3.connect('patients.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM medications_reference')
medications = cursor.fetchall()
for med in medications:
    print(f"\nMedication: {med[1]}")
    print(f"  Common Dosages: {med[2]}")
    print(f"  Usage: {med[3]}")
    print(f"  Side Effects: {med[4]}")
conn.close()

def get_patients_by_side_effect(side_effect_symptom):
    """
    Generate SQL query to get patients and their medications based on a side effect symptom.
    
    Args:
        side_effect_symptom: The symptom/side effect to search for (e.g., 'headache', 'nausea')
    
    Returns:
        tuple: (SQL query string, list of parameters)
    """
    query = """
    SELECT DISTINCT 
        p.patient_id,
        p.first_name,
        p.last_name,
        p.date_of_birth,
        pm.medication_name,
        pm.dosage,
        mr.side_effects
    FROM patients p
    INNER JOIN patient_medications pm ON p.patient_id = pm.patient_id
    INNER JOIN medications_reference mr ON pm.medication_name = mr.medication_name
    WHERE mr.side_effects LIKE ?
    ORDER BY p.last_name, p.first_name, pm.medication_name;
    """
    
    # Use wildcards for partial matching
    search_param = f"%{side_effect_symptom}%"
    
    return query, [search_param]


# Example usage: Find patients taking medications that may cause headaches
print("\n" + "="*100)
print("Patients taking medications with 'headache' as a side effect:")
print("="*100)

conn = sqlite3.connect('patients.db')
cursor = conn.cursor()

sql_query, params = get_patients_by_side_effect('headache')
cursor.execute(sql_query, params)
results = cursor.fetchall()

print(f"\nFound {len(results)} patient-medication combinations with 'headache' as a potential side effect:\n")
for row in results:
    print(f"Patient: {row[1]} {row[2]} (ID: {row[0]}, DOB: {row[3]})")
    print(f"  Medication: {row[4]} - {row[5]}")
    print(f"  Possible Side Effects: {row[6]}")
    print()

# Another example: Find patients taking medications that may cause nausea
print("\n" + "="*100)
print("Patients taking medications with 'nausea' as a side effect:")
print("="*100)

sql_query, params = get_patients_by_side_effect('nausea')
cursor.execute(sql_query, params)
results = cursor.fetchall()

print(f"\nFound {len(results)} patient-medication combinations with 'nausea' as a potential side effect:\n")
for row in results[:10]:  # Show first 10 results
    print(f"Patient: {row[1]} {row[2]} (ID: {row[0]})")
    print(f"  Medication: {row[4]} - {row[5]}")
    print()

conn.close()