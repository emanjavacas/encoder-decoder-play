
import dynet as dy
from utils import run_rnn, preprocess_seq, argmax, log_loss


class SimpleRNNNetwork(object):
    def __init__(self, depth, emb_dim, hid_dim, char2int, int2char):
        self.model = dy.Model()
        self.char2int = char2int
        self.int2char = int2char
        vocab_size = len(char2int)
        # embeddings
        self.lookup = self.model.add_lookup_parameters((vocab_size, emb_dim))
        # rnn
        self.rnn = dy.LSTMBuilder(depth, emb_dim, hid_dim, self.model)
        # output projection (from hid_dim -> vocab_size)
        self.out_W = self.model.add_parameters((vocab_size, hid_dim))
        self.out_b = self.model.add_parameters((vocab_size))

    def _embed_seq(self, in_seq):
        "embed a sequence"
        return [self.lookup[item] for item in in_seq]

    def _output_softmax(self, rnn_output):
        "computes output projection to get a softmax distribution"
        out_W = dy.parameter(self.out_W)
        out_b = dy.parameter(self.out_b)
        probs = dy.softmax(out_W * rnn_output + out_b)
        return probs

    def _probs(self, in_seq):
        "runs the rnn from the input string sequence to the output probs"
        embedded = self._embed_seq(in_seq)
        for rnn_output in run_rnn(embedded, self.rnn.initial_state()):
            probs = self._output_softmax(rnn_output)
            yield probs

    def get_loss(self, in_str, out_str, loss_fn=log_loss):
        "compute loss for a given pair of input/output sequences"
        in_seq = preprocess_seq(in_str, self.char2int)
        out_seq = preprocess_seq(out_str, self.char2int)
        dy.renew_cg()
        loss = []
        for probs, char in zip(self._probs(in_seq), out_seq):
            loss.append(loss_fn(probs, char))
        loss = dy.esum(loss)
        return loss

    def generate(self, in_str):
        dy.renew_cg()
        in_seq = preprocess_seq(in_str, self.char2int)
        output_str = ""
        for probs in self._probs(in_seq):
            char = argmax(probs)
            output_str += self.int2char[char]
        return output_str

    def log_checkpoint(self, idx, loss, val_set, target, trans=True, **kwargs):
        val_loss = sum(self.get_loss(X, y).value() for (X, y) in val_set)
        print('Example [%d], train loss [%3f], dev loss [%3f]' %
              (idx, loss, val_loss))
        if trans:
            pred = self.generate(target)
            print('source [%s] ==> translation [%s]' % (target, pred))

    def train(self, train_X, val_X, checkpoint=100,
              epochs=20, lr=0.1, decay=0.0, target='redrum', **kwargs):
        trainer = dy.SimpleSGDTrainer(self.model, lr, decay)

        for idx, (in_str, out_str) in enumerate(train_X * epochs):
            loss = self.get_loss(in_str, out_str)
            train_loss = loss.value()
            loss.backward()
            trainer.update()

            if idx > 0 and idx % checkpoint == 0:
                self.log_checkpoint(idx, train_loss, val_X, target, **kwargs)


class BiRNNNetwork(SimpleRNNNetwork):
    def __init__(self, depth, emb_dim, hid_dim, char2int, int2char):
        self.char2int = char2int
        self.int2char = int2char
        vocab_size = len(char2int)
        self.model = dy.Model()
        # embeddings
        self.lookup = self.model.add_lookup_parameters((vocab_size, emb_dim))
        # forward rn
        self.fwd_rnn = dy.LSTMBuilder(depth, emb_dim, hid_dim, self.model)
        # backward rnn
        self.bwd_rnn = dy.LSTMBuilder(depth, emb_dim, hid_dim, self.model)
        # output projection
        self.out_W = self.model.add_parameters((vocab_size, hid_dim * 2))
        self.out_b = self.model.add_parameters((vocab_size))

    def _probs(self, in_seq):
        embedded = self._embed_seq(in_seq)

        # rnn outputs
        fwd_outputs = run_rnn(embedded, self.fwd_rnn.initial_state())
        bwd_outputs = run_rnn(embedded[::-1],
                              self.bwd_rnn.initial_state())[::-1]

        for (fwd_output, bwd_output) in zip(fwd_outputs, bwd_outputs):
            rnn_output = dy.concatenate([fwd_output, bwd_output])
            probs = self._output_softmax(rnn_output)
            yield probs
