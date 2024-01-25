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

   model = wrp.v5.Model("/path/to/model.st", turbo=False)
   logits, state = wrp.v5.run_one(model, [114, 514], state=None)
   ```
   
# Advanced Usage
1. Move state to memory:
   
   ```python
   logits, state = wrp.v5.run_one(model, [114, 514], state=None) # returned state is on vram
   memory_state = state.back()
   ```
   
2. Load state from memory:
   
   ```python
   vram_state = wrp.v5.ModelState(model, 1)
   vram_state.load(memory_state)
   logits, state = wrp.v5.run_one(model, [114, 514], state=vram_state)
   ```
   
