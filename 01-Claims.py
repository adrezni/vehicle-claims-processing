import pandas as pd
import numpy
import markovify
import csv

df = pd.read_csv('response.csv')
df = df.fillna('')
df['response']=df.iloc[:,3]+df.iloc[:,5]+df.iloc[:,6]
df['issue'] = df.iloc[:,1]
df['symptom'] = df.iloc[:,2] + df.iloc[:,4]
subset = df.iloc[:,-3:]

# Function builds the model according to what issue (e.g. brakes, starter, other) is given
def train_markov_type(data, issue):
    return markovify.Text(data[data["issue"] == issue].response, retain_original=False, state_size=2)


# Function takes one of the 'issue' models and creates a randomly-generated sentence of length up to 200 characters.
# Note only creates '1' sentence
def make_sentence(model, length=100):
    return model.make_short_sentence(length, max_overlap_ratio = .7, max_overlap_total=15)


other_model = train_markov_type(subset, "Other")
brakes_model = train_markov_type(subset, "Brakes")
starter_model = train_markov_type(subset, "Starter")


# Generate sentences together with their label as an array of tuples.
# For example:  [(Sentence1, category), (Sentence2, category)...]
# parameter, models is an array of tuples, with each tuple containing a model and a category.
# For example: [(brake_model, 'brake'), (other_model, 'other'), (starter_model, 'starter')]
# Parameter, weights is an array of relative weights, for example: [14,7,7]
def generate_cases(models, weights=None):
    if weights is None:
        weights = [1] * len(models)
    choices = []  # Array of tuples of weight and models
    total_weight = float(sum(weights))

    for i in range(len(weights)):
        choices.append((float(sum(weights[0:i + 1])) / total_weight, models[i]))

    # Return a tuple of model and category that are randomly selected by given weights.
    def choose_model():
        r = numpy.random.uniform()
        for (model_weight, model) in choices:
            if r <= model_weight:
                return model
        return choices[-1][1]

    while True:
        local_model = choose_model()  # (<markovify.text.Text object at 0x7ff3dbf72a60>, 'other')
        # local_model[0]) is the markovify model, local_model[1] is the category
        yield make_sentence(local_model[0]), local_model[1]


generated_cases = generate_cases([(other_model,'other'), (brakes_model,'brakes'), (starter_model,'starter')], [14,7,7])
# Tuples with sentence and category
sentence_tuples = [next(generated_cases)  for i in range(100)]  # create 100 sentence/category tuples

# Write to csv file
with open('testdata1.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',', lineterminator='\n')
    writer.writerows(sentence_tuples)