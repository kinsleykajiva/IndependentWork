string = "Dissatisfied
Dissatisfied
Dissatisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Dissatisfied
Dissatisfied
Dissatisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Dissatisfied
Dissatisfied
Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Dissatisfied
Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Dissatisfied
Dissatisfied
Dissatisfied
Dissatisfied
Dissatisfied
Dissatisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Satisfied
Dissatisfied
Dissatisfied
Satisfied
Dissatisfied
Neither Satisfied Nor Dissatisfied
Dissatisfied
Neither Satisfied Nor Dissatisfied
Dissatisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied
Satisfied
Neither Satisfied Nor Dissatisfied
Neither Satisfied Nor Dissatisfied
Satisfied
Satisfied
Satisfied"

import io, pprint

line = buf.readline()
count = 0
current = ""
disagreements = 0

while line != "":
    if count % 3:
        current = line
    else:
        if line != current:
            disagreements = disagreements + 1
    count = count + 1
    line = buf.readline()

print disagreements
