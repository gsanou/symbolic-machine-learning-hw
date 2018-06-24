from itertools import product

from bridge import subsume
from logic import *
from dataset import Sample


class Taxonomy:
    def __init__(self, taxonomy: Set[Literal]):
        self.graph = self.build_graph(taxonomy)

    def cnp(self, a: Constant, b: Constant) -> Constant:
        if a == b:
            return a
        if a in self.graph:
            return self.cnp(b, self.graph[a])
        elif b in self.graph:
            return self.cnp(a, self.graph[b])
        else:
            return None

    def build_graph(self, taxonomy: Set[Literal]):
        graph: Dict[Term, Term] = {}
        for literal in taxonomy:
            graph[literal.atom.terms[0]] = literal.atom.terms[1]
        return graph


class NoneTaxonomy(Taxonomy):
    def __init__(self, taxonomy: Set[Literal] = None):
        pass

    def cnp(self, a: Constant, b: Constant) -> Constant:
        return None


class Substitution:
    i = 0

    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy
        self.substitution = {}

    def transform(self, key: Tuple[Term, Term]) -> Term:
        term1, term2 = key
        if isinstance(term1, CompoundTerm):
            key = (str(term1), term2)
        if isinstance(term2, CompoundTerm):
            key = (key[0], str(term2))

        if key not in self.substitution:
            if term1 != term2:
                self.substitution[key] = self.resolve_substitution(term1, term2)
            else:
                self.substitution[key] = term1

        return self.substitution[key]

    def resolve_substitution(self, t1: Term, t2: Term):
        if isinstance(t1, Constant) and isinstance(t2, Constant):
            if t1 == t2:
                return t1

            cnp = self.taxonomy.cnp(t1, t2)
            if cnp is not None:
                return cnp
            else:
                self.i += 1
                return Variable("V%d" % self.i)

        elif isinstance(t1, Constant) and isinstance(t2, Variable):
            return t2

        elif isinstance(t1, Variable) and isinstance(t2, Constant):
            return t1
        else:
            self.i += 1
            return Variable("V%d" % self.i)


class LGG:
    def __init__(self, taxonomy: Set[Literal] = None):
        self.taxonomy = taxonomy

    def apply(self, gamma_a: Clause, gamma_b: Clause) -> Clause:
        if self.taxonomy is None:
            substitution = Substitution(NoneTaxonomy())
        else:
            substitution = Substitution(Taxonomy(self.taxonomy))

        literals = self.iterate_table(substitution, gamma_a, gamma_b)
        return Clause(literals)

    def iterate_table(self, substitution: Substitution, gamma_a: Clause, gamma_b: Clause) -> Iterable[
        Literal]:
        for lit_a, lit_b in product(gamma_a, gamma_b):
            if self.is_comparable(lit_a, lit_b):
                terms = self.lgg_comparable_terms(substitution, lit_a, lit_b)
                literal = Literal(Atom(lit_a.atom.predicate, terms), lit_a.positive)
                yield literal

    @staticmethod
    def is_comparable(a: Literal, b: Literal) -> bool:
        return a.positive == b.positive and a.atom.predicate == b.atom.predicate

    def lgg_comparable_terms(self, substitution: Substitution, terms1: Iterable[Term],
                             terms2: Iterable[Term]) -> Iterable[Term]:
        """
        Computes LGG(terms1, terms2)
        """
        for term_a, term_b in zip(terms1, terms2):
            if isinstance(term_a, CompoundTerm) and isinstance(term_b, CompoundTerm) \
                    and term_a.functor == term_b.functor:
                yield CompoundTerm(term_a.functor, self.lgg_comparable_terms(substitution, term_a.terms, term_b.terms))
            else:
                yield substitution.transform((term_a, term_b))


class Reduction:
    def apply(self, gamma: Clause) -> Clause:
        literals: Set[Literal] = set(gamma.literals)
        for lit in gamma:
            if subsume(gamma, Clause(literals - {lit})):
                literals.remove(lit)
        return Clause(literals)


class LGGResolver:
    ''' 
    Your task is to implement the first-order generalization agent based on the LGG algorithm here, as discussed in the
    lecture (see 4.2 Generalization of Clauses in SMU textbook version of year 201/).

    The class represents an generalization agent of first-order clauses based on the LGG algorithm. He initially starts
    with no hypothesis at all. Each time he gets an observation (in form of Sample, consisting of a class and a clause, 
    by calling his method, seeObservation(Sample)), he should change his hypothesis accordingly; i.e. in the case that
    prediction of the observation by the agent's hypothesis differs from the sample class. Recall that the agent predict
    the observation as positive iff the observed clause is theta subsumed by the agent hypothesis. Also recall that we 
    assume that there is no noise in the data. 
    
    One can obtain current hypothesis of the agent (Clause) by calling getHypothesis().

    Your first task is to implement the agent/LGG algorithm here in the method seeObservation(Sample). 
    Your second task is to implement lgg with the clause reduction step, which is called by seeObservation(Sample,reduceClause=True).
    Your third task is to implement lgg with taxonomical extension. Taxonomical information about constants are given to 
    the agent by the class constructor, e.g. LGGResolver(taxonomical=info) where info is a set of literals of a form 
     'isa(dog,mammal)'. It is ensured that from this set a forest can be formed, i.e. set of rooted oriented trees. 
    '''

    def __init__(self, taxonomical: Set[Literal] = None):
        '''
        Constructs new LGGResolver.
        
        Parameter taxonomical contains set of literals describing taxonomical information about the domain. It either
        may be None, i.e. no taxonomy provided, or it consists of literal of pairs isa/2 hierarchy, e.g. isa(car, vehicle).
        It is always ensured that literals in the set describes a forest, i.e. set of rooted oriented trees.
        
        :type taxonomical : Set of Literal
        :rtype: LGGResolver
        '''
        self.taxonomical = taxonomical
        self.hypothesis: Clause = None

    def getHypothesis(self) -> Clause:
        '''
        Returns current hypothesis of the agent.
        
        :rtype: Clause 
        '''
        return self.hypothesis

    def seeObservation(self, sample: Sample, reduceClause: bool = False) -> None:
        '''
        Performs LGG with the current hypothesis stored in the agent iff the the sample has positive label but the agent does predict the opposite class for the sample given.
        
        If reduction is set to True, then the agent process also the reduction step. You do not have to implement the 
        whole functionality, i.e. subsumption engine. To test whether one clause subsumes another one, 
        e.g. \alpha \subseq_{\theta} \beta, use library method subsume from package logic, e.g. subsume(\alpha,\beta).   

        
        :type sample: Sample
        :type reduceClause : bool
        :rtype: None 
        '''
        lgg = LGG(self.taxonomical)
        reduction = Reduction()

        if sample.positiveClass:
            if self.hypothesis is None:
                self.hypothesis = sample.data
            elif not subsume(self.hypothesis, sample.data):
                self.hypothesis = lgg.apply(self.hypothesis, sample.data)
                print("Hypothesis changed to: {}".format(self.hypothesis))
                if reduceClause:
                    reductee = reduction.apply(self.hypothesis)
                    if reductee != self.hypothesis:
                        self.hypothesis = reductee
                        print("Hypothesis reduced to: {}".format(self.hypothesis))
                    else:
                        print("No reduction plausible.")
            else:
                print("Data is theta-subsumed by hypothesis.")
        else:
            print("Negative sample, thus ignored.")

        print("-" * 100)
