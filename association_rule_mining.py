#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)
print(itemset)
#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
encoded_vals = []
for index, row in df.iterrows():
    cur_row = dict()
    for item in itemset:
        if row.str.contains(item).sum() > 0:
            cur_row[item] = 1
        else:
            cur_row[item] = 0
    encoded_vals.append(cur_row)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:
for index, row in rules.iterrows():
    row_dict = dict(row)
    
    #printing antecedents
    ants = set(row_dict['antecedents'])
    for index, item in enumerate(ants):
        print(f'{item}', end='')
        if index < (len(ants) - 1):
            print(', ', end='')

    print(' -> ', end='')
    #print consequents
    cons = set(row_dict['consequents'])
    for index, item in enumerate(cons):
        print(f'{item}', end='')
        if index < (len(cons) - 1):
            print(', ', end='')
    
    #print support
    sup = row_dict['antecedent support']
    conf = row_dict['confidence']
    #prior = row_dict['consequent support']
    supportCount = 0
    for trans in encoded_vals:
        has_cons = True
        for item in cons:
            if trans[item] == 0:
                has_cons = False
        
        if has_cons:
            supportCount += 1

    prior = supportCount/len(encoded_vals)
    #print newline
    print()
    print(f"Support: {sup}")
    print(f"Confidence: {conf}")
    print(f"Prior: {prior}")
    print("Gain in Confidence: " + str(100*(conf-prior)/prior))


    #printing newline
    print()

#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
#see above

#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#see above

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()