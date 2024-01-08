
import argparse
import json
import math 
import time
from pathlib import Path 
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim 
import numpy as np 
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from model import LoRALinear, Model, ModelArgs
from sentencepiece import SentencePieceProcessor

def build_parser():
    parser = argparse.ArgumentParser(
        description="LoRA training script")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="A path to model files"
    )

    # Generation Tasks
    parser.add_argument(
        "--num-tokens",
        '-n',
        type=int,
        default=100,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--write-every", type=int, default=1, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt for generation",
        default=None,
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=25,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Adam learning rate."
    )
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=200,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        default=None,
        help="Load path to resume training with the given adapter weights.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Save/load path for the trained adapter weights.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    
    return parser


class Tokenizer:
    def __init__(self, model_path:str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "_"
        assert self._model.vocab_size() == self._model.get_piece_size()


    def encode(self, s: str, eos: bool = False):
        toks = [self._model.bos_id(), *self._model.encode(s)]
        if eos:
            toks.append(self._model.eos_id())
        return toks
    
    @property
    def eos_id(self):
        return self._model.eos_id()
    
    def decode(self, toks: List[int]):
        out = self._model.decode(toks)
        if toks and self._model.id_to_piece(toks[0])[0] == self._sep:
            return " " + out 
        return out
    
    @property
    def vocab_size(self):
        return self._model.vocab_size()
    

class Dataset:

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            raise ValueError(f"Path {path} does not exist")
        else: 
            with open(path, "r") as f:
                self._data = [json.loads(l) for l in f]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]
    
    def __len__(self):
        return len(self._data)
    

def load(args):
    names = ("train", "valid", "test")
    train, valid, test = (Dataset(Path(args.data) / f"{n}.jsonl") for n in names)

    if args.train and len(train) == 0:
        raise ValueError("No training data found")
    if args.train and len(valid) == 0:
        raise ValueError("No validation data found")
    if args.test and len(test) == 0:
        raise ValueError("No test data found")
    return train, valid, test

def loss(model, inputs, targets, lengths):

    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # mask padding tokens
    length_mask = mx.arrange(inputs.shape[1])[None, : ] < lengths[:, None]

    # compute the loss
    ce = nn.losses.cross_entropy(logits,targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(dataset, tokenizer, batch_size, train=False):

    while True:
        indices = np.arange(len(dataset))

        if train:
            np.random.shuffle(indices)

        # Collect batches from dataset 
        for i in range(0, len(indices) - batch_size + 1, batch_size):

            # Encode 
            batch = [
                tokenizer.encode(dataset[i + idx], eos=True) 
                for idx in range(batch_size)
            ]
            lengths = np.array([len(b) for b in batch])


            if max(lengths) > 2048:
                print(
                    f"Skipping batch of length {max(lengths)} because it is too long"
                )
                
            # Pad to max_length
            batch_arr = np.zeros((batch_size, max(lengths)), dtype=np.int32)
            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch = mx.array(batch_arr)
            yield batch[:, :-1], batch[:, 1:], lengths - 1 # inputs[0 to -1), targets[1 to -1], lengths - 1

        if not train:
            break


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):

    all_losses = []
    n_tokens = 0

    for idx, batch in zip(range(num_batches), iterate_batches(dataset, tokenizer, batch_size)):

        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        n_tokens += toks

    return np.sum(all_losses) / n_tokens

def train(model, train_set, val_set, optimizer, loss, tokenizer, args):

    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0

    # Main Training
    start = time.perf_counter()
    for idx, batch in zip(range(args.iters), iterate_batches(train_set, tokenizer, args.batch_size, train=True)):

        (loss_value, toks), grads = loss_value_and_grad(model, *batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss_value)

        # Record loss
        losses.append(loss_value.item())
        n_tokens += toks.item()


    if (idx + 1) %  args.steps_per_report == 0:
        train_loss = np.mean(losses)

        stop = time.perf_counter()
        print(
                f"Iter {idx + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {args.steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
        losses = []
        n_tokens = 0
        start = time.perf_counter()

    if idx == 0 or (idx + 1) % args.steps_per_eval == 0:
       stop = time.perf_counter()
       val_loss = evaluate(
                model, val_set, loss, tokenizer, args.batch_size, args.val_batches
            )
       print(
                f"Iter {idx + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )
       start = time.perf_counter()


def generate(model, prompt, tokenizer, args):

    print(args.prompt, end="", flush=True)
    prompt = mx.array(tokenizer.encode(args.prompt))

    def generate_step():
        temp = args.temp

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1/ temp))
        
        logits, cache = model(prompt[None])
        y = sample(logits[:, -1, :])
        yield y

        while True:
            logits, cache = model(y[:, None], cache)
            y = sample(logits.squeeze(1))
            yield y

    tokens = []
    for token, _ in zip(generate_step(), range(args.num_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)

def load_model(folder: str):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.pop("quantization", None)
        model_args = ModelArgs(**config)
    model = Model(model_args)
    if quantization is not None:
        quantization["linear_class_predicate"] = lambda m: isinstance(
            m, nn.Linear
        ) and (m.weight.shape[0] != model_args.vocab_size)
        nn.QuantizedLinear.quantize_module(model, **quantization)

    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))
    model.update(weights)
    return model, tokenizer