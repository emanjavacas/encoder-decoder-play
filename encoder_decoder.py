

import numpy as np
import dynet as dy

import utils as u
from rnn import SimpleRNNNetwork


class EncoderDecoderNetwork(SimpleRNNNetwork):
    def __init__(self, depth, emb_dim, hid_dim,
                 char2int, int2char, builder=dy.LSTMBuilder, add_pred=False):
        enc_depth, dec_depth = depth
        enc_hid_dim, dec_hid_dim = hid_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.add_pred = add_pred
        self.char2int = char2int
        self.int2char = int2char
        vocab_size = len(char2int)

        self.model = dy.Model()
        # embeddings
        self.lookup = self.model.add_lookup_parameters((vocab_size, emb_dim))
        # rnns
        self.enc_rnn = builder(enc_depth, emb_dim, enc_hid_dim, self.model)
        # decoder input has same dimension as encoder output (ev. + embedding)
        if self.add_pred:
            dec_in = enc_hid_dim + emb_dim
        else:
            dec_in = enc_hid_dim
        self.dec_rnn = builder(dec_depth, dec_in, dec_hid_dim, self.model)
        # output_projection
        self.out_W = self.model.add_parameters((vocab_size, dec_hid_dim))
        self.out_b = self.model.add_parameters((vocab_size))

    def encode(self, embedded):
        # use only last hidden state
        return u.run_rnn(embedded, self.enc_rnn.initial_state())[-1]

    def recur(self, state, encoded, last_char_emb):
        """compute input to next state given the current decoder state, the encoded seq
        (output of the encoding network), and the embedding of the last char"""
        if self.add_pred:
            return dy.concatenate([encoded, last_char_emb])
        else:
            return encoded

    def make_decoder(self, in_seq, **kwargs):
        embedded = self._embed_seq(in_seq)
        encoded = self.encode(embedded)
        state_vec = dy.vecInput(self.enc_hid_dim)
        # EOS as zero-vector for 1st step
        last_char_emb = self.lookup[self.char2int[u.EOS]]
        if self.add_pred:
            init = dy.concatenate([state_vec, last_char_emb])
        else:
            init = state_vec
        s = self.dec_rnn.initial_state().add_input(init)
        while True:
            # create a new decoding state (s) combining previous decoding step,
            # encoded seq (output of encoder) and ev. last input encoding
            new_state = self.recur(s, encoded, last_char_emb)
            s = s.add_input(new_state)
            yield self._output_softmax(s.output())
            last_char = yield
            last_char_emb = self.lookup[last_char]

    def get_loss(self, in_str, out_str, loss_fn=u.log_loss):
        in_seq = u.preprocess_seq(in_str, self.char2int)
        out_seq = u.preprocess_seq(out_str, self.char2int)
        dy.renew_cg()
        loss = []
        decoder = self.make_decoder(in_seq)
        probs = next(decoder)
        for out_char in out_seq:
            loss.append(loss_fn(probs, out_char))
            next(decoder)
            probs = decoder.send(out_char)
        loss = dy.esum(loss)
        return loss

    def generate(self, in_str, max_len=2, **kwargs):
        dy.renew_cg()
        in_seq = u.preprocess_seq(in_str, self.char2int)
        out_str, pred, EOSint = "", None, self.char2int[u.EOS]
        decoder = self.make_decoder(in_seq, **kwargs)
        probs = next(decoder)
        while pred != EOSint and len(out_str) < (max_len * len(in_seq)):
            pred = u.argmax(probs)  # greedy search (take best prediction)
            next(decoder)
            probs = decoder.send(pred)
            out_str += self.int2char[pred]
        return out_str


class AttentionNetwork(EncoderDecoderNetwork):
    def __init__(self, depth, emb_dim, hid_dim, att_dim, char2int, int2char,
                 add_pred=True, **kwargs):
        super(AttentionNetwork, self).\
            __init__(depth, emb_dim, hid_dim,
                     char2int, int2char, add_pred=add_pred, **kwargs)
        # attention weights
        enc_hid_dim, dec_hid_dim = hid_dim
        self.enc2att = self.model.add_parameters((att_dim, enc_hid_dim))
        self.dec2att = self.model.add_parameters((att_dim, dec_hid_dim))
        self.att_v = self.model.add_parameters((1, att_dim))

        self.char2int = char2int
        self.int2char = int2char
        self.current_weights = []

    def get_attention_matrix(self):
        if not self.current_weights:
            raise ValueError('Empty attention matrix, pass store_weights=True')
        W = np.array(self.current_weights)
        self.reset_attention_matrix()
        return W

    def reset_attention_matrix(self):
        self.current_weights = []

    def attend(self, enc_h_ts_mat, dec_h_t, encatt, store_weights=False):
        """
        Parameters:
        -----------
        enc_h_ts_mat: dynet.Expression, (seq_len x enc_hid_dim)
            matrix of encoding hidden state column vectors
        dec_h_t: dynet.RNNState, (dec_hid_dim)
            current decoder hidden state
        encatt: dynet.Expression, (seq_len x att_dim)
            projection of the encoder hidden states into the attention space
        store_weights: bool,
            whether to store attention weights
        """
        dec2att = dy.parameter(self.dec2att)
        att_v = dy.parameter(self.att_v)
        # project output of last hidden layer (state.h()[-1] == state.output())
        # to the dimensionality of the attention space
        decatt = dec2att * dec_h_t.output()
        # projection vector att_v
        # unnormalized var-len alignment vector (with len == source seq len)
        # (seq_len)
        unnormalized_weights = att_v * dy.tanh(dy.colwise_add(encatt, decatt))
        weights = dy.softmax(dy.transpose(unnormalized_weights))
        if store_weights:
            self.current_weights.append(weights.value())
        context = enc_h_ts_mat * weights
        return context

    def checkpoint(self, e, idx, tr_loss, val_loss, target='abcd', trans=True,
                   prefix='attention', plot=True):
        super(AttentionNetwork, self) \
            .checkpoint(e, idx, tr_loss, val_loss, target=target, trans=trans)
        if plot:
            import matplotlib.pyplot as plt
            from hinton_diagram import hinton
            pred = self.generate(target, store_weights=True)
            fig = hinton(self.get_attention_matrix(),
                         xlabels=list(target),
                         ylabels=list(pred.replace(u.EOS, '')))
            plt.savefig('./imgs/%s-%d.png' % (prefix, idx))
            plt.close(fig)

    def encode(self, embedded):
        # return full encoded sequence
        return u.run_rnn(embedded, self.enc_rnn.initial_state())

    def recur(self, s, enc_mat, last_char_emb, encatt, store_weights=False):
        """compute input to next state given the current decoder state, the encoded seq
        (output of the encoding network), and the embedding of the last char

        Parameters:
        -----------
        s: dy.RNNState,
            current decoder state
        enc_mat: dy.Expression,
            see self.attend
        last_char_emb: dy.Expression,
            vector corresponding to the last decoded character embedding
        encatt: dy.Expression,
            see self.attend
        """
        context = self.attend(enc_mat, s, encatt, store_weights=store_weights)
        if self.add_pred:
            return dy.concatenate([context, last_char_emb])
        else:
            return context

    def make_decoder(self, in_seq, **kwargs):
        """
        Creates a decoder generator to be used as co-routine to the decoding
        procedure. It has two steps (i) it first output a probability dist
        on the vocabulary and (ii) it accepts a symbol to be fedback into
        the generation of the next symbol.
        See self.generate and self.
        """
        embedded = self._embed_seq(in_seq)
        enc_mat = dy.concatenate_cols(self.encode(embedded))
        # variables to compute and cache the encoder projection onto att space
        enc2att = dy.parameter(self.enc2att)
        encatt = None
        # EOS as zero-vector for 1st step
        last_char_emb = self.lookup[self.char2int[u.EOS]]
        # init hidden state of decoder should take last encoding hidden state
        state_vec = dy.vecInput(self.enc_hid_dim)
        if self.add_pred:
            init = dy.concatenate([state_vec, last_char_emb])
        else:
            init = state_vec
        s = self.dec_rnn.initial_state().add_input(init)
        while True:
            # (maybe) project encoding hidden seq onto attention space
            encatt = encatt or enc2att * enc_mat
            # create a new decoding state (s) combining previous decoding step,
            # encoded seq (output of encoder) and ev. last input encoding
            new_state = self.recur(s, enc_mat, last_char_emb, encatt, **kwargs)
            s = s.add_input(new_state)
            # TODO: according to Bahdanau 2015, the new state is computed with
            # deep output + single maxout:
            # p(y_i | s_i, y_{i-1}, c_i) \prop exp(y_i^T * W_o * t_i)
            # where $t_i = [max(t^~_{i, 2j-1}, t^~_{i, 2j})]^T_{j=1,...,l}$
            # and $t^~_i = U_o * s_{i-1} + V_o * Ey_{i-1} * C_o * c_i$
            yield self._output_softmax(s.output())
            last_char = yield
            last_char_emb = self.lookup[last_char]
