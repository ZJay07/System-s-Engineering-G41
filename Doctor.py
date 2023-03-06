print("Please answer the following yes/no questions:")

questions = {
    "Medical history": "past illnesses, surgeries, or medical procedures",
    "Medications": "currently taking any medications",
    "Allergies": "allergies to medications or other substances",
    "Family history": "family history of medical conditions or illnesses",
    "Lifestyle factors": "lifestyle habits that may affect your health",
    "Vital signs": "had your vital signs measured recently, including blood pressure, heart rate, respiratory rate, and temperature",
    "Lab results": "had any recent lab tests or imaging studies, such as blood tests, urine tests, X-rays, or MRIs",
    "Progress notes": "had any recent medical appointments or hospital stays",
    "Physical exam": "had a physical exam recently"
}

details = {}

for question, detail in questions.items():
    answer = input(f"Do you have {detail}? ").lower()
    if answer == "yes":
        details[question] = input(f"Please provide more information about your {detail}: ")
    else:
        details[question] = f"no {detail}"

paragraph = "\n".join([f"{question}: {detail}" for question, detail in details.items()])

print(paragraph)
