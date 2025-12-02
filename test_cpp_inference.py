# test_cpp_inference.py
# Script para probar la inferencia C++ con imágenes reales de MNIST

import torch
from torchvision import datasets, transforms
import numpy as np
import struct
import os

def save_mnist_samples(num_samples=10):
    """Guarda algunas imágenes de MNIST para probar C++"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    os.makedirs('test_images', exist_ok=True)
    
    print(f"Guardando {num_samples} imágenes de prueba...")
    
    for i in range(num_samples):
        image, label = test_dataset[i]
        image_np = image.numpy().flatten()
        
        # Guardar como binario
        filename = f'test_images/mnist_{i}_label_{label}.bin'
        with open(filename, 'wb') as f:
            f.write(struct.pack(f'{len(image_np)}f', *image_np))
        
        print(f"  {filename} - Label: {label}")
    
    print(f"\n✓ {num_samples} imágenes guardadas en test_images/")

def test_python_model(model_path='vit_mnist_best.pth'):
    """Prueba el modelo Python para comparar"""
    
    # Importar el modelo
    import sys
    sys.path.append('.')
    
    # Cargar modelo
    from train_vit import VisionTransformer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Cargar dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    print("\nPredicciones Python (primeras 10 imágenes):")
    print("=" * 50)
    
    correct = 0
    for i in range(10):
        image, label = test_dataset[i]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(dim=1).item()
        
        is_correct = "✓" if pred == label else "✗"
        print(f"  Imagen {i}: Predicción={pred}, Real={label} {is_correct}")
        
        if pred == label:
            correct += 1
    
    print(f"\nPrecisión en 10 muestras: {correct}/10 = {100*correct/10:.1f}%")

def create_cpp_test_program():
    """Crea un programa C++ mejorado para cargar imágenes binarias"""
    
    cpp_code = """// test_mnist.cpp - Programa para probar con imágenes reales
// Compilar: g++ -std=c++17 -O3 test_mnist.cpp -o test_mnist
// Requiere: vit_inference.cpp compilado como librería o incluido

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

vector<float> load_mnist_binary(const string& filepath) {
    ifstream file(filepath, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: no se puede abrir " << filepath << endl;
        return {};
    }
    
    vector<float> image(28 * 28);
    file.read(reinterpret_cast<char*>(image.data()), image.size() * sizeof(float));
    
    return image;
}

int extract_label_from_filename(const string& filename) {
    // Formato: mnist_X_label_Y.bin
    size_t pos = filename.find("_label_");
    if (pos == string::npos) return -1;
    
    string label_str = filename.substr(pos + 7);
    label_str = label_str.substr(0, label_str.find("."));
    
    return stoi(label_str);
}

int main() {
    cout << "======================================" << endl;
    cout << "Test Vision Transformer - MNIST Real" << endl;
    cout << "======================================" << endl;
    
    // Aquí incluirías tu clase VisionTransformer
    // VisionTransformer model;
    // model.load_weights("model_weights.json");
    
    string test_dir = "test_images";
    int correct = 0;
    int total = 0;
    
    cout << "\\nCargando imágenes de " << test_dir << "..." << endl;
    
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.path().extension() != ".bin") continue;
        
        string filename = entry.path().filename().string();
        int true_label = extract_label_from_filename(filename);
        
        if (true_label == -1) continue;
        
        vector<float> image = load_mnist_binary(entry.path().string());
        
        // int predicted = model.predict(image);
        int predicted = 0;  // Placeholder
        
        total++;
        if (predicted == true_label) correct++;
        
        string status = (predicted == true_label) ? "✓" : "✗";
        cout << "  " << filename << ": Pred=" << predicted 
             << ", Real=" << true_label << " " << status << endl;
    }
    
    cout << "\\n======================================" << endl;
    cout << "Precisión: " << correct << "/" << total 
         << " = " << (100.0 * correct / total) << "%" << endl;
    cout << "======================================" << endl;
    
    return 0;
}
"""
    
    with open('test_mnist.cpp', 'w') as f:
        f.write(cpp_code)
    
    print("\n✓ Archivo test_mnist.cpp creado")

def compare_outputs():
    """Compara las salidas de Python vs C++"""
    print("\n" + "="*50)
    print("COMPARACIÓN PYTHON vs C++")
    print("="*50)
    
    print("\n1. Primero ejecuta: python test_cpp_inference.py")
    print("   Esto guardará imágenes de prueba y mostrará predicciones Python")
    
    print("\n2. Luego recompila C++ con la versión actualizada:")
    print("   g++ -std=c++17 -O3 vit_inference.cpp -o vit_inference")
    
    print("\n3. Ejecuta: ./vit_inference")
    print("   Debería dar predicciones correctas (no solo 0)")
    
    print("\n4. Para probar con imágenes reales:")
    print("   g++ -std=c++17 -O3 test_mnist.cpp -o test_mnist")
    print("   ./test_mnist")

if __name__ == '__main__':
    import sys
    
    print("Vision Transformer - Test Suite")
    print("=" * 50)
    
    # Guardar imágenes de prueba
    save_mnist_samples(10)
    
    # Probar modelo Python
    if os.path.exists('vit_mnist_best.pth'):
        test_python_model()
    else:
        print("\n⚠ No se encontró vit_mnist_best.pth")
        print("  Ejecuta primero: python train_vit.py")
    
    # Crear programa de test C++
    create_cpp_test_program()
    
    # Mostrar instrucciones
    compare_outputs()