print("Please answer the following yes/no questions:")

# Medical history
med_history = input("Do you have any past illnesses, surgeries, or medical procedures? ").lower()
if med_history == "yes":
    med_history_details = input("Please provide more information: ")
else:
    med_history_details = "no history of past illnesses, surgeries, or medical procedures"

# Medications
medications = input("Are you currently taking any medications? ").lower()
if medications == "yes":
    medications_details = input("Please provide more information: ")
else:
    medications_details = "not currently taking any medications"

# Allergies
allergies = input("Do you have any allergies to medications or other substances? ").lower()
if allergies == "yes":
    allergies_details = input("Please provide more information: ")
else:
    allergies_details = "no allergies to medications or other substances"

# Family history
family_history = input("Do you have a family history of medical conditions or illnesses? ").lower()
if family_history == "yes":
    family_history_details = input("Please provide more information: ")
else:
    family_history_details = "no family history of medical conditions or illnesses"

# Lifestyle factors
lifestyle_factors = input("Do you have any lifestyle habits that may affect your health, such as diet, exercise, smoking, or alcohol consumption? ").lower()
if lifestyle_factors == "yes":
    lifestyle_factors_details = input("Please provide more information: ")
else:
    lifestyle_factors_details = "no lifestyle habits that may affect your health"

# Vital signs
vital_signs = input("Have you had your vital signs measured recently, including blood pressure, heart rate, respiratory rate, and temperature? ").lower()
if vital_signs == "yes":
    vital_signs_details = input("Please provide the results: ")
else:
    vital_signs_details = "no recent measurements of vital signs"

# Lab results
lab_results = input("Have you had any recent lab tests or imaging studies, such as blood tests, urine tests, X-rays, or MRIs? ").lower()
if lab_results == "yes":
    lab_results_details = input("Please provide more information: ")
else:
    lab_results_details = "no recent lab tests or imaging studies"

# Progress notes
progress_notes = input("Have you had any recent medical appointments or hospital stays? ").lower()
if progress_notes == "yes":
    progress_notes_details = input("Please provide more information: ")
else:
    progress_notes_details = "no recent medical appointments or hospital stays"

# Physical exam
physical_exam = input("Have you had a physical exam recently? ").lower()
if physical_exam == "yes":
    physical_exam_details = input("Please provide the results: ")
else:
    physical_exam_details = "no recent physical exam"

# Generate paragraph
paragraph = f"Medical history: {med_history_details}. \nMedications: {medications_details}. \nAllergies: {allergies_details}. \nFamily history: {family_history_details}. \nLifestyle factors: {lifestyle_factors_details}. \nVital signs: {vital_signs_details}. \nLab results: {lab_results_details}. \nProgress notes: {progress_notes_details}. \nPhysical exam: {physical_exam_details}."

# Print paragraph
print(paragraph)
