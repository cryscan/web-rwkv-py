import web_rwkv_py as wrp

model = wrp.Model("./assets/models/RWKV-x060-World-1B6-v2-20240208-ctx4096.st")
tokenizer = wrp.Tokenizer("./assets/rwkv_vocab_v20230424.json")

prompt = "The Eiffel Tower is"
tokens = tokenizer.encode(prompt)
output, state = model.run(tokens)
print(prompt, end="")

# We must explicitly call this to clone the state.
state_1 = model.clone_state(state)

prompt = " located in the city of"
tokens = tokenizer.encode(prompt)
print(prompt, end="")
output, state = model.run(tokens, state)
output_token, _ = max(enumerate(output), key=lambda x: x[1])

prompt = " located in the city of"
tokens = tokenizer.encode(prompt)
output, _ = model.run_full(tokens, state_1)
output = output[-1]
output_token_1, _ = max(enumerate(output), key=lambda x: x[1])

str = str(bytearray(tokenizer.decode([output_token_1])), encoding='utf-8')
print(str)

assert(output_token == output_token_1)
