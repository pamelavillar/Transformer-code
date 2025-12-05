#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace std;

class Tensor {
public:
  vector<float> datos;
  vector<int> forma;

  Tensor() {}
  Tensor(vector<int> f) : forma(f) {
    int tam = 1;
    for (int dim : f)
      tam *= dim;
    datos.resize(tam, 0.0f);
  }

  int tamano() const {
    int t = 1;
    for (int dim : forma)
      t *= dim;
    return t;
  }

  float &operator()(int i, int j) { return datos[i * forma[1] + j]; }

  const float &operator()(int i, int j) const {
    return datos[i * forma[1] + j];
  }

  float &at(int i, int j, int k) {
    return datos[i * forma[1] * forma[2] + j * forma[2] + k];
  }
};

Tensor multiplicar(const Tensor &A, const Tensor &B) {
  int m = A.forma[0], k = A.forma[1], n = B.forma[1];
  Tensor C({m, n});

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float suma = 0.0f;
      for (int p = 0; p < k; p++) {
        suma += A(i, p) * B(p, j);
      }
      C(i, j) = suma;
    }
  }
  return C;
}

Tensor transponer(const Tensor &A) {
  Tensor B({A.forma[1], A.forma[0]});
  for (int i = 0; i < A.forma[0]; i++) {
    for (int j = 0; j < A.forma[1]; j++) {
      B(j, i) = A(i, j);
    }
  }
  return B;
}

void sumar_sesgo(Tensor &x, const vector<float> &sesgo) {
  for (int i = 0; i < x.forma[0]; i++) {
    for (int j = 0; j < x.forma[1]; j++) {
      x(i, j) += sesgo[j];
    }
  }
}

void softmax_filas(Tensor &x) {
  for (int i = 0; i < x.forma[0]; i++) {
    float max_val = x(i, 0);
    for (int j = 1; j < x.forma[1]; j++) {
      max_val = max(max_val, x(i, j));
    }

    float suma = 0.0f;
    for (int j = 0; j < x.forma[1]; j++) {
      x(i, j) = exp(x(i, j) - max_val);
      suma += x(i, j);
    }

    for (int j = 0; j < x.forma[1]; j++) {
      x(i, j) /= suma;
    }
  }
}

void gelu(Tensor &x) {
  for (auto &val : x.datos) {
    val = 0.5f * val *
          (1.0f + tanh(sqrt(2.0f / M_PI) * (val + 0.044715f * pow(val, 3))));
  }
}

void normalizacion_capa(Tensor &x, const vector<float> &gamma,
                        const vector<float> &beta, float eps = 1e-5f) {
  int long_sec = x.forma[0];
  int dim = x.forma[1];

  for (int i = 0; i < long_sec; i++) {
    float media = 0.0f;
    for (int j = 0; j < dim; j++) {
      media += x(i, j);
    }
    media /= dim;

    float varianza = 0.0f;
    for (int j = 0; j < dim; j++) {
      float dif = x(i, j) - media;
      varianza += dif * dif;
    }
    varianza /= dim;

    float inv_std = 1.0f / sqrt(varianza + eps);
    for (int j = 0; j < dim; j++) {
      x(i, j) = gamma[j] * (x(i, j) - media) * inv_std + beta[j];
    }
  }
}

class VisionTransformer {
private:
  int tam_imagen = 28;
  int tam_parche = 4;
  int dim_incrustacion = 64;
  int num_cabezas = 4;
  int profundidad = 4;
  int num_clases = 10;
  int num_parches;

  map<string, vector<float>> pesos;

  Tensor reformar_peso(const vector<float> &w, vector<int> forma) {
    Tensor t(forma);
    t.datos = w;
    return t;
  }

  Tensor atencion_multicabeza(Tensor &x, int idx_bloque) {
    string prefijo = "blocks." + to_string(idx_bloque) + ".attn.";

    int long_sec = x.forma[0];
    int d = x.forma[1];
    int dim_cabeza = d / num_cabezas;

    Tensor peso_qkv = reformar_peso(pesos[prefijo + "qkv.weight"], {3 * d, d});
    Tensor peso_qkv_T = transponer(peso_qkv);
    vector<float> &sesgo_qkv = pesos[prefijo + "qkv.bias"];

    Tensor qkv = multiplicar(x, peso_qkv_T);
    sumar_sesgo(qkv, sesgo_qkv);

    Tensor q({long_sec, d}), k({long_sec, d}), v({long_sec, d});

    for (int i = 0; i < long_sec; i++) {
      for (int j = 0; j < d; j++) {
        q(i, j) = qkv(i, j);         // primeros d elementos
        k(i, j) = qkv(i, j + d);     // siguientes d elementos
        v(i, j) = qkv(i, j + 2 * d); // últimos d elementos
      }
    }

    Tensor salida({long_sec, d});
    salida.datos.assign(salida.tamano(), 0.0f);

    for (int cabeza = 0; cabeza < num_cabezas; cabeza++) {

      Tensor puntajes_atencion({long_sec, long_sec});
      float escala = 1.0f / sqrt((float)dim_cabeza);

      for (int i = 0; i < long_sec; i++) {
        for (int j = 0; j < long_sec; j++) {
          float suma = 0.0f;
          for (int idx_dim = 0; idx_dim < dim_cabeza; idx_dim++) {
            int idx_q = cabeza * dim_cabeza + idx_dim;
            int idx_k = cabeza * dim_cabeza + idx_dim;
            suma += q(i, idx_q) * k(j, idx_k);
          }
          puntajes_atencion(i, j) = suma * escala;
        }
      }

      softmax_filas(puntajes_atencion);

      // Atención * V
      for (int i = 0; i < long_sec; i++) {
        for (int idx_dim = 0; idx_dim < dim_cabeza; idx_dim++) {
          float suma = 0.0f;
          for (int j = 0; j < long_sec; j++) {
            int idx_v = cabeza * dim_cabeza + idx_dim;
            suma += puntajes_atencion(i, j) * v(j, idx_v);
          }
          salida(i, cabeza * dim_cabeza + idx_dim) = suma;
        }
      }
    }

    Tensor peso_proy = reformar_peso(pesos[prefijo + "proj.weight"], {d, d});
    Tensor peso_proy_T = transponer(peso_proy);
    vector<float> &sesgo_proy = pesos[prefijo + "proj.bias"];

    Tensor salida_final = multiplicar(salida, peso_proy_T);
    sumar_sesgo(salida_final, sesgo_proy);

    return salida_final;
  }

  Tensor red_mlp(Tensor &x, int idx_bloque) {
    string prefijo = "blocks." + to_string(idx_bloque) + ".mlp.";

    int d = x.forma[1];
    int dim_oculta_mlp = d * 4;

    Tensor peso_fc1 =
        reformar_peso(pesos[prefijo + "0.weight"], {dim_oculta_mlp, d});
    Tensor peso_fc1_T = transponer(peso_fc1);
    vector<float> &sesgo_fc1 = pesos[prefijo + "0.bias"];

    Tensor h = multiplicar(x, peso_fc1_T);
    sumar_sesgo(h, sesgo_fc1);
    gelu(h);

    Tensor peso_fc2 =
        reformar_peso(pesos[prefijo + "2.weight"], {d, dim_oculta_mlp});
    Tensor peso_fc2_T = transponer(peso_fc2);
    vector<float> &sesgo_fc2 = pesos[prefijo + "2.bias"];

    Tensor salida = multiplicar(h, peso_fc2_T);
    sumar_sesgo(salida, sesgo_fc2);

    return salida;
  }

public:
  VisionTransformer() {
    num_parches = (tam_imagen / tam_parche) * (tam_imagen / tam_parche);
  }

  bool cargar_pesos(const string &ruta) {
    ifstream archivo(ruta);
    if (!archivo.is_open()) {
      cerr << "Error: no se puede abrir " << ruta << endl;
      return false;
    }

    json j;
    archivo >> j;

    auto config = j["config"];
    dim_incrustacion = config["embed_dim"];
    profundidad = config["depth"];
    num_cabezas = config["num_heads"];

    function<vector<float>(const json &)> aplanar =
        [&](const json &arr) -> vector<float> {
      vector<float> resultado;
      if (arr.is_number()) {
        resultado.push_back(arr.get<float>());
      } else if (arr.is_array()) {
        for (const auto &item : arr) {
          auto sub = aplanar(item);
          resultado.insert(resultado.end(), sub.begin(), sub.end());
        }
      }
      return resultado;
    };

    for (auto &[clave, valor] : j["weights"].items()) {
      pesos[clave] = aplanar(valor);
    }

    return true;
  }

  Tensor incrustacion_parches(const vector<float> &imagen) {
    Tensor parches({num_parches, dim_incrustacion});

    vector<float> &peso_conv = pesos["patch_embed.proj.weight"];
    vector<float> &sesgo_conv = pesos["patch_embed.proj.bias"];

    int parches_por_lado = tam_imagen / tam_parche;

    for (int fila_parche = 0; fila_parche < parches_por_lado; fila_parche++) {
      for (int col_parche = 0; col_parche < parches_por_lado; col_parche++) {
        int idx_parche = fila_parche * parches_por_lado + col_parche;

        for (int canal_salida = 0; canal_salida < dim_incrustacion;
             canal_salida++) {
          float suma = sesgo_conv[canal_salida];

          for (int i = 0; i < tam_parche; i++) {
            for (int j = 0; j < tam_parche; j++) {
              int fila_img = fila_parche * tam_parche + i;
              int col_img = col_parche * tam_parche + j;
              int idx_img = fila_img * tam_imagen + col_img;

              int idx_peso =
                  canal_salida * (tam_parche * tam_parche) + i * tam_parche + j;

              suma += imagen[idx_img] * peso_conv[idx_peso];
            }
          }

          parches(idx_parche, canal_salida) = suma;
        }
      }
    }

    return parches;
  }

  vector<float> propagar(const vector<float> &imagen) {
    Tensor parches = incrustacion_parches(imagen);

    vector<float> &token_cls = pesos["cls_token"];
    Tensor x({num_parches + 1, dim_incrustacion});

    for (int j = 0; j < dim_incrustacion; j++) {
      x(0, j) = token_cls[j];
    }

    for (int i = 0; i < num_parches; i++) {
      for (int j = 0; j < dim_incrustacion; j++) {
        x(i + 1, j) = parches(i, j);
      }
    }

    vector<float> &incrust_pos = pesos["pos_embed"];
    for (int i = 0; i < num_parches + 1; i++) {
      for (int j = 0; j < dim_incrustacion; j++) {
        x(i, j) += incrust_pos[i * dim_incrustacion + j];
      }
    }

    for (int idx_bloque = 0; idx_bloque < profundidad; idx_bloque++) {
      string prefijo = "blocks." + to_string(idx_bloque) + ".";

      Tensor x_norm = x;
      normalizacion_capa(x_norm, pesos[prefijo + "norm1.weight"],
                         pesos[prefijo + "norm1.bias"]);

      Tensor salida_atn = atencion_multicabeza(x_norm, idx_bloque);
      for (int i = 0; i < x.tamano(); i++) {
        x.datos[i] += salida_atn.datos[i];
      }

      x_norm = x;
      normalizacion_capa(x_norm, pesos[prefijo + "norm2.weight"],
                         pesos[prefijo + "norm2.bias"]);

      Tensor salida_mlp = red_mlp(x_norm, idx_bloque);
      for (int i = 0; i < x.tamano(); i++) {
        x.datos[i] += salida_mlp.datos[i];
      }
    }

    normalizacion_capa(x, pesos["norm.weight"], pesos["norm.bias"]);

    vector<float> salida_cls(dim_incrustacion);
    for (int j = 0; j < dim_incrustacion; j++) {
      salida_cls[j] = x(0, j);
    }

    Tensor peso_cabeza =
        reformar_peso(pesos["head.weight"], {num_clases, dim_incrustacion});
    Tensor peso_cabeza_T = transponer(peso_cabeza);
    vector<float> &sesgo_cabeza = pesos["head.bias"];

    vector<float> logits(num_clases, 0.0f);
    for (int i = 0; i < num_clases; i++) {
      float suma = sesgo_cabeza[i];
      for (int j = 0; j < dim_incrustacion; j++) {
        suma += salida_cls[j] * peso_cabeza_T(j, i);
      }
      logits[i] = suma;
    }

    return logits;
  }

  int predecir(const vector<float> &imagen) {
    auto logits = propagar(imagen);
    /*cout << "\nLogits: ";
    for (int i = 0; i < logits.size(); i++) {
        cout << i << "=" << logits[i] << " ";
    }
    cout << endl;*/
    return max_element(logits.begin(), logits.end()) - logits.begin();
  }
};