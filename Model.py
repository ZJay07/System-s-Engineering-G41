import re
print("Please choose your preferences below: ")
preferenceDict ={}

questions = [
    "yes or yeh",
    "no or nah",
    "want to or wanna",
    "going to or gonna"
]

for i in range(0,len(questions)):
    response = input(questions[i] + " (1/2) ")
    splitquestion = questions[i].split(" or ")
    currentkey = splitquestion[0] + " Preference"
    if response == "1":
        preferenceDict[currentkey] = splitquestion[0]
    elif response == "2":
        preferenceDict[currentkey] = splitquestion[1]
        
def identify_question_type(question):
    # Define a regular expression to match yes/no/maybe questions
    yes_no_re = r'^\s*(can|could|should|would|will|is|am|are|was|were|do|does|did|have|has|had)\s*.*\?$'

    # Check if the question matches the regular expression
    if re.match(yes_no_re, question):
        return "yes/no/maybe"
    else:
        return "other"         

def provide_options(question_type):
    if question_type == "yes/no/maybe":
        yesPreference = preferenceDict.get("yes Preference")
        noPreference = preferenceDict.get("no Preference")
        return f"Options: {yesPreference}, {noPreference}, Maybe"
    else:
        return "Sorry, no options available for this type of question."

# Example usage:
question = "How are you feeling today?"
question = question.lower()
question_type = identify_question_type(question)
