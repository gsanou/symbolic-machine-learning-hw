from bridge import subsume
from logic import Clause

clause = Clause.parse("!female(ingrid),!female(jana),!parent(ingrid, jana),daugther(jana)")
hypothesis = Clause.parse("!female(V1), !female(V3), !parent(V2, V1), daugther(V1)")
hypothesis2 = Clause.parse("!female(V1), !parent(V2, V1), daugther(V1)")
clause2 = Clause.parse("daugther(X),!parent(Y,X),!human(Y),!female(X)")
print()

print(clause)
print(hypothesis)
print(clause2)
print(subsume(hypothesis, clause))
print(subsume(hypothesis, clause2))
print(subsume(hypothesis, hypothesis))
print(subsume(hypothesis, hypothesis2))
