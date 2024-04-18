# Web-RWKV-Py
Python binding for [`web-rwkv`](https://github.com/cryscan/web-rwkv).

# Todos
- [x] Basic V5 inference support
- [x] Support V4, V5 and V6
- [ ] Batched inference

# Usage
1. Install python and rust.
2. Install maturin by
   
   ```bash
   $ pip install maturin
   ```
4. Build and install:

   ```bash
   $ maturin develop --release
   ```

5. Try using `web-rwkv` in python:

   ```python
   import web_rwkv_py as wrp

   model = wrp.Model(
      "/path/to/model.st", # model path
      quant=0,             # int8 quantization layers
      quant_nf4=0,         # nf4 quantization layers
   )
   logits, state = model.run([114, 514], state=None)
   ```
   
# Advanced Usage
1. Clone states:

   ```python
   logits, state = model.run([114, 514], state=None)
   state_cloned = model.clone_state(state)
   ```
   
2. Return predictions of all tokens (not only the last's):

   ```python
   logits, state = model.run_full([114, 514, 1919, 810], state=None)
   assert(len(logits) == 4)
   ```

3. Specify token chunk size (default is 128, the larger the faster prefilling is, but you will need a beefy GPU):

   ```python
   tokens = [114, 514, 1919, 810]
   logits, state = model.run_full(tokens, state=None, token_chunk_size=256)
   ```
