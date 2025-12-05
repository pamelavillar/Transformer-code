# Trabajo final

Transformer que aprende el data set MNIST

# Pasos

1. Correr train_vit.py (aqui hace todo el transformar y va guardar los pesos generados)
2. Correr mnist_gen (aqui se van a pasar las imágenes a formato bin)
3. Compilamos g++ -std=c++17 -O3 test_mnist_float.cpp -o test_float
   ./test_float

test_mnist_float usa vit_inference.cpp

# integrantes

- Rodrigo Silva
- Pamela Villar
