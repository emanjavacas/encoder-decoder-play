
import dynet as dy

EOS = '<EOS>'


# generator functions
def identity(string):
    return string, string


def reverse(string):
    return string, string[::-1]


def ntuple(string, n=3):
    return string, ''.join([char * n for char in string])


def double(string):
    return ntuple(string, n=2)


def reversedouble(string):
    return string, ''.join([char + char for char in string[::-1]])


def skipchar(string, skip=1):
    splitby = skip + 1
    return string, ''.join([string[i::splitby] for i in range(splitby)])


# general logic functions
def preprocess_seq(string, char2int):
    "add EOS at the end of each string"
    return [char2int[c] for c in list(string) + [EOS]]


def run_rnn(input_vecs, init_state):
    return [s.output() for s in init_state.add_inputs(input_vecs)]


def log_loss(out_probs, true_target):
    return -dy.log(dy.pick(out_probs, true_target))


def argmax(vec):
    vec = vec.vec_value()
    return vec.index(max(vec))
