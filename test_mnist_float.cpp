#include "vit_inference.cpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

vector<float> cargar_mnist_binario(const string &filepath) {
  ifstream file(filepath, ios::binary);
  if (!file.is_open()) {
    cerr << "Error: no se puede abrir " << filepath << endl;
    return {};
  }

  vector<float> imagen(28 * 28);
  file.read(reinterpret_cast<char *>(imagen.data()),
            imagen.size() * sizeof(float));

  return imagen;
}

int extraer_de_archivo(const string &filename) {
  size_t pos = filename.find("_label_");
  if (pos == string::npos)
    return -1;
  string label_str = filename.substr(pos + 7);
  label_str = label_str.substr(0, label_str.find("."));
  return stoi(label_str);
}

int main() {
  cout << "Test Vision Transformer - MNIST Real" << endl;

  VisionTransformer model;
  if (!model.cargar_pesos("model_weights.json")) {
    return 1;
  }

  string test_dir = "test_images";
  int correct = 0;
  int total = 0;

  cout << "Cargando imágenes de " << test_dir << "..." << endl;

  for (const auto &entry : fs::directory_iterator(test_dir)) {
    if (entry.path().extension() != ".bin")
      continue;

    string filename = entry.path().filename().string();
    int true_label = extraer_de_archivo(filename);
    if (true_label == -1)
      continue;

    vector<float> image = cargar_mnist_binario(entry.path().string());

    if (total == 0) {
      cout << "  Primeros 5 píxeles: ";
      for (int i = 0; i < 5; i++)
        cout << image[i] << " ";
      cout << endl;
    }

    int predicted = model.predecir(image);

    total++;
    if (predicted == true_label)
      correct++;

    string status = (predicted == true_label) ? "✓" : "✗";
    cout << "  " << filename << ": Pred=" << predicted
         << ", Real=" << true_label << " " << status << endl;
  }

  cout << "Precisión: " << correct << "/" << total << " = "
       << (100.0 * correct / total) << "%" << endl;

  return 0;
}