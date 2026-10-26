"""Pad a drafter GGUF's vocabulary to the target's, so llama.cpp accepts it for speculative decoding.

  pad_draft.py TARGET.gguf DRAFT.gguf OUT.gguf

Nanbeige4.1-3B and Nanbeige4.3-33B share a tokenizer, but the 33B's GGUF pads it to 166,912 entries and the 3B's
to 166,144. llama.cpp refuses a drafter whose vocabulary differs by more than 128 entries. This copies the
drafter with the target's tokenizer metadata and zero rows in token_embd/output for the extra padding ids. The
padding ids never appear in real text, and the target checks every drafted token, so the output is unchanged.
"""
import sys

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter

VOCAB_TENSORS = ("token_embd.weight", "output.weight")


def value(field):
    kind = field.types[0]
    if kind == GGUFValueType.ARRAY:
        if field.types[-1] == GGUFValueType.STRING:
            return [bytes(field.parts[i]).decode("utf-8", errors="surrogateescape") for i in field.data]
        return [field.parts[i].tolist()[0] for i in field.data]
    if kind == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode()
    return field.parts[field.data[0]].tolist()[0]


def copy_field(writer: GGUFWriter, name: str, field) -> None:
    kind = field.types[0]
    if kind == GGUFValueType.ARRAY and field.types[-1] == GGUFValueType.STRING:
        writer.add_array(name, value(field))
    elif kind == GGUFValueType.ARRAY:  # keep the element type: scores float32, token types int32
        writer.add_key_value(name, value(field), kind, sub_type=field.types[-1])
    else:
        writer.add_key_value(name, value(field), kind)


def main(target_path: str, draft_path: str, out_path: str) -> None:
    target, draft = GGUFReader(target_path), GGUFReader(draft_path)
    tokens_tgt = value(target.fields["tokenizer.ggml.tokens"])
    tokens_dft = value(draft.fields["tokenizer.ggml.tokens"])
    n_tgt, n_dft = len(tokens_tgt), len(tokens_dft)
    if n_tgt < n_dft or tokens_tgt[:n_dft] != tokens_dft:
        sys.exit("the drafter's tokens are not a prefix of the target's: not the same tokenizer")
    print(f"vocab {n_dft:,} -> {n_tgt:,}")

    writer = GGUFWriter(out_path, arch=value(draft.fields["general.architecture"]))
    for name, field in draft.fields.items():
        if not (name.startswith(("GGUF.", "tokenizer.")) or name == "general.architecture"):
            copy_field(writer, name, field)
    for name, field in target.fields.items():
        if name.startswith("tokenizer."):
            copy_field(writer, name, field)

    for tensor in draft.tensors:
        data = np.asarray(tensor.data)  # quantized tensors come as one row of blocks per vocab entry
        if tensor.name in VOCAB_TENSORS and int(tensor.shape[-1]) == n_dft:
            data = np.concatenate([data, np.zeros((n_tgt - n_dft,) + data.shape[1:], dtype=data.dtype)])
            print(f"{tensor.name} ({tensor.tensor_type.name}): {n_dft:,} -> {data.shape[0]:,} rows")
        writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("wrote", out_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
