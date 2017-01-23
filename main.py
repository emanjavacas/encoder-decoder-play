
from random import choice, randrange
import utils as u
from rnn import SimpleRNNNetwork, BiRNNNetwork
from encoder_decoder import EncoderDecoderNetwork, AttentionNetwork


if __name__ == '__main__':

    import random
    random.seed(1001)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('-t', '--train_len', default=3000, type=int)
    parser.add_argument('-v', '--val_len', default=50, type=int)
    parser.add_argument('-m', '--min_len', default=1, type=int)
    parser.add_argument('-M', '--max_len', default=15, type=int)
    parser.add_argument('-f', '--sample_fn', default='reverse', type=str)
    parser.add_argument('-d', '--depth', default=1, type=int)
    parser.add_argument('-e', '--emb_dim', default=4, type=int)
    parser.add_argument('-H', '--hid_dim', default=124, type=int)
    parser.add_argument('-a', '--att_dim', default=64, type=int)
    parser.add_argument('-E', '--epochs', default=5, type=int)
    parser.add_argument('-p', '--prefix', default='model', type=str)
    parser.add_argument('-V', '--vocab', default='abcd', type=str)
    parser.add_argument('-g', '--target', default='redrum', type=str)
    parser.add_argument('-T', '--test_length', action='store_true')
    parser.add_argument('-c', '--checkpoint', default=500, type=int)
    parser.add_argument('-P', '--plot', action='store_true')
    args = parser.parse_args()

    if args.sample_fn == 'identity':
        SAMPLE_FN = u.identity
    elif args.sample_fn == 'reverse':
        SAMPLE_FN = u.reverse
    elif args.sample_fn == 'double':
        SAMPLE_FN = u.double
    elif args.sample_fn == 'triple':
        SAMPLE_FN = lambda x: u.ntuple(x, n=3)
    elif args.sample_fn == 'quadruple':
        SAMPLE_FN = lambda x: u.ntuple(x, n=4)
    elif args.sample_fn == 'quintuple':
        SAMPLE_FN = lambda x: u.ntuple(x, n=5)
    elif args.sample_fn == 'reversedouble':
        SAMPLE_FN = u.reversedouble
    elif args.sample_fn == 'skipchar1':
        SAMPLE_FN = lambda x: u.skipchar(x, skip=1)
    elif args.sample_fn == 'skipchar2':
        SAMPLE_FN = lambda x: u.skipchar(x, skip=2)
    elif args.sample_fn == 'skipchar3':
        SAMPLE_FN = lambda x: u.skipchar(x, skip=3)
    else:
        raise ValueError('non exisiting fn [%s]' % args.sample_fn)

    TRAIN_LEN = args.train_len
    VAL_LEN = args.val_len
    MIN_LEN = args.min_len
    MAX_LEN = args.max_len
    EPOCHS = args.epochs
    VOCAB = list(set(args.vocab).union(set(args.target)))
    VOCAB.append(u.EOS)

    int2char = list(VOCAB)
    char2int = {c: i for i, c in enumerate(VOCAB)}

    DEPTH = args.depth
    EMB_DIM = args.emb_dim
    HID_DIM = args.hid_dim
    ATT_DIM = args.att_dim

    # training set generators
    def generate_str(min_len, max_len, vocab):
        randlen = randrange(min_len, max_len)
        return ''.join([choice(vocab[:-1]) for _ in range(randlen)])

    def generate_set(size, vocab, min_len=1, max_len=15, sample_fn=u.reverse):
        return [sample_fn(generate_str(min_len, max_len, vocab))
                for _ in range(size)]

    train_set = generate_set(TRAIN_LEN, VOCAB,
                             sample_fn=SAMPLE_FN,
                             min_len=MIN_LEN, max_len=MAX_LEN)
    val_set = generate_set(VAL_LEN, VOCAB,
                           sample_fn=SAMPLE_FN,
                           min_len=MIN_LEN, max_len=MAX_LEN)

    if args.model == 'SimpleRNNNetwork':
        rnn = SimpleRNNNetwork(DEPTH, EMB_DIM, HID_DIM, char2int, int2char)

    elif args.model == 'BiRNNNetwork':
        rnn = BiRNNNetwork(DEPTH, EMB_DIM, HID_DIM, char2int, int2char)

    elif args.model == 'EncoderDecoderNetwork':
        rnn = EncoderDecoderNetwork((DEPTH, DEPTH),
                                    EMB_DIM, (HID_DIM, HID_DIM),
                                    char2int, int2char)

    elif args.model == 'AttentionNetwork':
        rnn = AttentionNetwork((DEPTH, DEPTH), EMB_DIM,
                               (HID_DIM, HID_DIM), ATT_DIM,
                               char2int, int2char)

    else:
        raise ValueError('non existing model [%s]' % args.model)

    try:
        rnn.train(train_set,
                  val_set,
                  epochs=EPOCHS,
                  target=args.target,
                  prefix=args.prefix,
                  checkpoint=args.checkpoint,
                  plot=args.plot)

    except KeyboardInterrupt:
        pass

    finally:
        if args.test_length:
            print("Testing model for different length buckets")
            ss = generate_set(100, VOCAB, 1, 5, sample_fn=SAMPLE_FN)
            ms = generate_set(100, VOCAB, 5, 10, sample_fn=SAMPLE_FN)
            ls = generate_set(100, VOCAB, 10, 15, sample_fn=SAMPLE_FN)

            def count_matches(rnn, val_set):
                matches = [rnn.generate(in_str) == out_str.replace(u.EOS, '')
                           for in_str, out_str in val_set]
                return matches.count(True)

            print("Shorts [%s]" % count_matches(rnn, ss))
            print("Medium [%s]" % count_matches(rnn, ms))
            print("Long [%s]" % count_matches(rnn, ls))
