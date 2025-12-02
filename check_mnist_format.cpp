#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
using namespace std;

int main() {
    // Leer primer archivo
    ifstream file("test_images/mnist_0_label_7.bin", ios::binary);
    
    // Obtener tamaño
    file.seekg(0, ios::end);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    
    cout << "Tamaño del archivo: " << size << " bytes" << endl;
    
    if (size == 28*28) {
        // Son uint8 (0-255)
        vector<uint8_t> data(28*28);
        file.read(reinterpret_cast<char*>(data.data()), 28*28);
        
        cout << "Formato: uint8_t (0-255)" << endl;
        cout << "Primeros 10 valores: ";
        for (int i = 0; i < 10; i++) {
            cout << (int)data[i] << " ";
        }
        cout << endl;
        
        float min_val = 255, max_val = 0;
        for (auto v : data) {
            min_val = min(min_val, (float)v);
            max_val = max(max_val, (float)v);
        }
        cout << "Rango: [" << min_val << ", " << max_val << "]" << endl;
        
    } else if (size == 28*28*4) {
        // Son float (ya normalizados?)
        vector<float> data(28*28);
        file.read(reinterpret_cast<char*>(data.data()), 28*28*4);
        
        cout << "Formato: float32" << endl;
        cout << "Primeros 10 valores: ";
        for (int i = 0; i < 10; i++) {
            cout << data[i] << " ";
        }
        cout << endl;
        
        float min_val = 1e9, max_val = -1e9;
        for (auto v : data) {
            min_val = min(min_val, v);
            max_val = max(max_val, v);
        }
        cout << "Rango: [" << min_val << ", " << max_val << "]" << endl;
    } else {
        cout << "Formato desconocido!" << endl;
    }
    
    return 0;
}