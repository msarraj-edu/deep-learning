
import numpy as np

def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''

    R = np.random.randint(1, window_size + 1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = set(words[start:idx] + words[idx + 1:stop + 1])

    return list(target_words)

def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''

    n_batches = len(words ) //batch_size

    # only full batches
    words = words[: n_batches *batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx: idx +batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x ] *len(batch_y))
        yield x, y

fn = get_batches([x for x in range(10)],batch_size=2,window_size=1)
for x_,y_ in fn:
    print('x:',x_,'y:',y_)


















