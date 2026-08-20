from layer import Layer

layer_hidden = Layer(2, 5)
layer_output = Layer(5, 1)

dados_xor = [
#  entradas  | saída esperada
    ([0, 0],    [0]),
    ([0, 1],    [1]),
    ([1, 0],    [1]),
    ([1, 1],    [0]),
]

learnRate = 0.5


print("=== Forward pass (rede ainda sem treino) ===")
entry = dados_xor[1][0]
expected = dados_xor[1][1]

hidden_output = layer_hidden.process(entry)
final_output = layer_output.process(hidden_output)
print(f"Entrada: {entry} | Esperado: {expected} | Saida da rede: {final_output}")

first_output = final_output

def test_learning_single_entry(steps):
    for i in range(steps):
        print(f"\n=== passo {i + 1} ===")
        entry, expected = dados_xor[1]  # [0,1] -> 1
        hidden_output = layer_hidden.process(entry)
        final_output = layer_output.process(hidden_output)
        print(f"Saida antes do último ajuste: {final_output}")

        output_deltas = layer_output.backward_output(expected, learnRate)
        print(f"Deltas da camada de saida: {output_deltas}")

        hidden_output = layer_hidden.process(entry)
        final_output = layer_output.process(hidden_output)
        print(f"Saida apos {i+1} ajuste(s): {final_output}")



test_learning_single_entry(100)

print(f"Saída antes do treino: {first_output}")
print(f"resultado esperado: {expected}")

