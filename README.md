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
   $ maturin develop
   ```

5. Try using `web-rwkv` in python:

   ```python
   import web_rwkv_py as wrp

   model = wrp.v5.Model(
      "/path/to/model.st", # model path
      quant=0,             # int8 quantization layers
      quant_nf4=0,         # nf4 quantization layers
      turbo=True,          # faster when reading long prompts
      token_chunk_size=256 # maximum tokens in an inference chunk (can be 32, 64, 256, 1024, etc.)
   )
   logits, state = wrp.v5.run_one(model, [114, 514], state=None)
   ```
   
# Advanced Usage
1. Move state to host memory:
   
   ```python
   logits, state = wrp.v5.run_one(model, [114, 514], state=None) # returned state is on vram
   state_cpu = state.back()
   ```
   
2. Load state from host memory:
   
   ```python
   state = wrp.v5.ModelState(model, 1)
   state.load(state_cpu)
   logits, state = wrp.v5.run_one(model, [114, 514], state=state_cpu)
   ```
   
3. Return predictions of all tokens (not only the last's):

   ```python
   logits, state = wrp.v5.run_one_full(model, [114, 514], state=None)
   len(logits) # 2
   ```
