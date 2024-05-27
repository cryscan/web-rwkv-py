import web_rwkv_py as wrp

model = wrp.Model("../ai00_rwkv_server/assets/models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.st")
tokenizer = wrp.Tokenizer("./assets/rwkv_vocab_v20230424.json")

prompt_0 = "The Eiffel Tower is"
tokens = tokenizer.encode(prompt_0)
output = model.run(tokens)

# We must explicitly call this to get the current state.
state = model.back_state()
state_1 = model.back_state(wrp.StateDevice.Gpu)

prompt = " located in the city of"
tokens = tokenizer.encode(prompt)
output = model.run(tokens)
output_token, _ = max(enumerate(output), key=lambda x: x[1])
word = str(bytearray(tokenizer.decode([output_token])), encoding='utf-8')
print(prompt_0 + prompt + word)

model.load_state(state)
output = model.run_full(tokens)
output = output[-65536:]
output_token_1, _ = max(enumerate(output), key=lambda x: x[1])
word = str(bytearray(tokenizer.decode([output_token_1])), encoding='utf-8')
print(prompt_0 + prompt + word)

model.load_state(state_1)
output = model.run(tokens)
output_token_2, _ = max(enumerate(output), key=lambda x: x[1])
word = str(bytearray(tokenizer.decode([output_token_2])), encoding='utf-8')
print(prompt_0 + prompt + word)

model.clear_state()
prompt = "The Space Needle is located in downtown"
tokens = tokenizer.encode(prompt)
output = model.run(tokens)
output_token_3, _ = max(enumerate(output), key=lambda x: x[1])
word = str(bytearray(tokenizer.decode([output_token_3])), encoding='utf-8')
print(prompt + word)

assert(output_token == output_token_1)
assert(output_token == output_token_2)
