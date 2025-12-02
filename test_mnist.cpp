#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include "vit_inference.cpp"   // <-- Agregar esto

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

    // Normalizar (si tu .bin no está normalizado aún)
    for (float &px : image)
        px = (px / 255.0f - 0.1307f) / 0.3081f;

    return image;
}

int extract_label_from_filename(const string& filename) {
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
    
    // ---------------------------
    // 1. Crear modelo
    // ---------------------------
    VisionTransformer model;
    if (!model.load_weights("model_weights.json")) {
        cerr << "Error al cargar pesos." << endl;
        return 1;
    }

    string test_dir = "test_images";
    int correct = 0;
    int total = 0;

    cout << "\nCargando imágenes de " << test_dir << "..." << endl;

    // ---------------------------
    // 2. Probar todas las imágenes
    // ---------------------------
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.path().extension() != ".bin") continue;

        string filename = entry.path().filename().string();
        int true_label = extract_label_from_filename(filename);

        if (true_label == -1) continue;

        vector<float> image = load_mnist_binary(entry.path().string());

        // 3. Predicción real
        int predicted = model.predict(image);

        // 4. Contabilizar accuracy
        total++;
        if (predicted == true_label) correct++;

        string status = (predicted == true_label) ? "✓" : "✗";
        cout << "  " << filename << ": Pred=" << predicted 
             << ", Real=" << true_label << " " << status << endl;
    }

    cout << "\n======================================" << endl;
    cout << "Precisión: " << correct << "/" << total 
         << " = " << (100.0 * correct / total) << "%" << endl;
    cout << "======================================" << endl;

    return 0;
}