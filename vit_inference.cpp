
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
  vector<float> data;
  vector<int> shape;

  Tensor() {}
  Tensor(vector<int> s) : shape(s) {
    int size = 1;
    for (int dim : s)
      size *= dim;
    data.resize(size, 0.0f);
  }

  int size() const {
    int s = 1;
    for (int dim : shape)
      s *= dim;
    return s;
  }

  float &operator()(int i, int j) { return data[i * shape[1] + j]; }

  const float &operator()(int i, int j) const { return data[i * shape[1] + j]; }

  float &at(int i, int j, int k) {
    return data[i * shape[1] * shape[2] + j * shape[2] + k];
  }
};

Tensor multipl(const Tensor &A, const Tensor &B) {
  int m = A.shape[0], k = A.shape[1], n = B.shape[1];
  Tensor C({m, n});

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int p = 0; p < k; p++) {
        sum += A(i, p) * B(p, j);
      }
      C(i, j) = sum;
    }
  }
  return C;
}

Tensor transpose(const Tensor &A) {
  Tensor B({A.shape[1], A.shape[0]});
  for (int i = 0; i < A.shape[0]; i++) {
    for (int j = 0; j < A.shape[1]; j++) {
      B(j, i) = A(i, j);
    }
  }
  return B;
}

void add_bias(Tensor &x, const vector<float> &bias) {
  for (int i = 0; i < x.shape[0]; i++) {
    for (int j = 0; j < x.shape[1]; j++) {
      x(i, j) += bias[j];
    }
  }
}

void softmax_rows(Tensor &x) {
  for (int i = 0; i < x.shape[0]; i++) {
    float max_val = x(i, 0);
    for (int j = 1; j < x.shape[1]; j++) {
      max_val = max(max_val, x(i, j));
    }

    float sum = 0.0f;
    for (int j = 0; j < x.shape[1]; j++) {
      x(i, j) = exp(x(i, j) - max_val);
      sum += x(i, j);
    }

    for (int j = 0; j < x.shape[1]; j++) {
      x(i, j) /= sum;
    }
  }
}

void gelu(Tensor &x) {
  for (auto &val : x.data) {
    val = 0.5f * val *
          (1.0f + tanh(sqrt(2.0f / M_PI) * (val + 0.044715f * pow(val, 3))));
  }
}

void layer_norm(Tensor &x, const vector<float> &gamma,
                const vector<float> &beta, float eps = 1e-5f) {
  int seq_len = x.shape[0];
  int dim = x.shape[1];

  for (int i = 0; i < seq_len; i++) {
    float mean = 0.0f;
    for (int j = 0; j < dim; j++) {
      mean += x(i, j);
    }
    mean /= dim;

    float var = 0.0f;
    for (int j = 0; j < dim; j++) {
      float diff = x(i, j) - mean;
      var += diff * diff;
    }
    var /= dim;

    float std_inv = 1.0f / sqrt(var + eps);
    for (int j = 0; j < dim; j++) {
      x(i, j) = gamma[j] * (x(i, j) - mean) * std_inv + beta[j];
    }
  }
}

// ============================================
// VISION TRANSFORMER
// ============================================

class VisionTransformer {
private:
  int img_size = 28;
  int patch_size = 4;
  int embed_dim = 64;
  int num_heads = 4;
  int depth = 4;
  int num_classes = 10;
  int num_patches;

  map<string, vector<float>> weights;

  Tensor reshape_weight(const vector<float> &w, vector<int> shape) {
    Tensor t(shape);
    t.data = w;
    return t;
  }

  Tensor multihead_attention(Tensor &x, int block_idx) {
    string prefix = "blocks." + to_string(block_idx) + ".attn.";

    int seq_len = x.shape[0];
    int d = x.shape[1];
    int head_dim = d / num_heads;

    Tensor qkv_weight =
        reshape_weight(weights[prefix + "qkv.weight"], {3 * d, d});
    Tensor qkv_weight_T = transpose(qkv_weight);
    vector<float> &qkv_bias = weights[prefix + "qkv.bias"];

    Tensor qkv = multipl(x, qkv_weight_T);
    add_bias(qkv, qkv_bias);

    Tensor q({seq_len, d}), k({seq_len, d}), v({seq_len, d});

    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < d; j++) {
        q(i, j) = qkv(i, j);         // primeros d elementos
        k(i, j) = qkv(i, j + d);     // siguientes d elementos
        v(i, j) = qkv(i, j + 2 * d); // últimos d elementos
      }
    }

    Tensor output({seq_len, d});
    output.data.assign(output.size(), 0.0f);

    for (int h = 0; h < num_heads; h++) {

      Tensor attn_scores({seq_len, seq_len});
      float scale = 1.0f / sqrt((float)head_dim);

      for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
          float sum = 0.0f;
          for (int d_idx = 0; d_idx < head_dim; d_idx++) {
            int q_idx = h * head_dim + d_idx;
            int k_idx = h * head_dim + d_idx;
            sum += q(i, q_idx) * k(j, k_idx);
          }
          attn_scores(i, j) = sum * scale;
        }
      }

      softmax_rows(attn_scores);

      // Attn * V
      for (int i = 0; i < seq_len; i++) {
        for (int d_idx = 0; d_idx < head_dim; d_idx++) {
          float sum = 0.0f;
          for (int j = 0; j < seq_len; j++) {
            int v_idx = h * head_dim + d_idx;
            sum += attn_scores(i, j) * v(j, v_idx);
          }
          output(i, h * head_dim + d_idx) = sum;
        }
      }
    }

    Tensor proj_weight =
        reshape_weight(weights[prefix + "proj.weight"], {d, d});
    Tensor proj_weight_T = transpose(proj_weight);
    vector<float> &proj_bias = weights[prefix + "proj.bias"];

    Tensor out = multipl(output, proj_weight_T);
    add_bias(out, proj_bias);

    return out;
  }

  Tensor mlp(Tensor &x, int block_idx) {
    string prefix = "blocks." + to_string(block_idx) + ".mlp.";

    int d = x.shape[1];
    int mlp_hidden = d * 4;

    Tensor fc1_weight =
        reshape_weight(weights[prefix + "0.weight"], {mlp_hidden, d});
    Tensor fc1_weight_T = transpose(fc1_weight);
    vector<float> &fc1_bias = weights[prefix + "0.bias"];

    Tensor h = multipl(x, fc1_weight_T);
    add_bias(h, fc1_bias);
    gelu(h);

    Tensor fc2_weight =
        reshape_weight(weights[prefix + "2.weight"], {d, mlp_hidden});
    Tensor fc2_weight_T = transpose(fc2_weight);
    vector<float> &fc2_bias = weights[prefix + "2.bias"];

    Tensor out = multipl(h, fc2_weight_T);
    add_bias(out, fc2_bias);

    return out;
  }

public:
  VisionTransformer() {
    num_patches = (img_size / patch_size) * (img_size / patch_size);
  }

  bool load_weights(const string &filepath) {
    ifstream file(filepath);
    if (!file.is_open()) {
      cerr << "Error: no se puede abrir " << filepath << endl;
      return false;
    }

    json j;
    file >> j;

    auto config = j["config"];
    embed_dim = config["embed_dim"];
    depth = config["depth"];
    num_heads = config["num_heads"];

    function<vector<float>(const json &)> flatten =
        [&](const json &arr) -> vector<float> {
      vector<float> result;
      if (arr.is_number()) {
        result.push_back(arr.get<float>());
      } else if (arr.is_array()) {
        for (const auto &item : arr) {
          auto sub = flatten(item);
          result.insert(result.end(), sub.begin(), sub.end());
        }
      }
      return result;
    };

    for (auto &[key, value] : j["weights"].items()) {
      weights[key] = flatten(value);
    }

    return true;
  }

  Tensor patch_embedding(const vector<float> &image) {
    Tensor patches({num_patches, embed_dim});

    vector<float> &conv_weight = weights["patch_embed.proj.weight"];
    vector<float> &conv_bias = weights["patch_embed.proj.bias"];

    int n_patches_side = img_size / patch_size;

    for (int p_row = 0; p_row < n_patches_side; p_row++) {
      for (int p_col = 0; p_col < n_patches_side; p_col++) {
        int patch_idx = p_row * n_patches_side + p_col;

        for (int out_ch = 0; out_ch < embed_dim; out_ch++) {
          float sum = conv_bias[out_ch];

          for (int i = 0; i < patch_size; i++) {
            for (int j = 0; j < patch_size; j++) {
              int img_row = p_row * patch_size + i;
              int img_col = p_col * patch_size + j;
              int img_idx = img_row * img_size + img_col;

              int weight_idx =
                  out_ch * (patch_size * patch_size) + i * patch_size + j;

              sum += image[img_idx] * conv_weight[weight_idx];
            }
          }

          patches(patch_idx, out_ch) = sum;
        }
      }
    }

    return patches;
  }

  vector<float> forward(const vector<float> &image) {
    Tensor patches = patch_embedding(image);

    vector<float> &cls_token_data = weights["cls_token"];
    Tensor x({num_patches + 1, embed_dim});

    for (int j = 0; j < embed_dim; j++) {
      x(0, j) = cls_token_data[j];
    }

    for (int i = 0; i < num_patches; i++) {
      for (int j = 0; j < embed_dim; j++) {
        x(i + 1, j) = patches(i, j);
      }
    }

    vector<float> &pos_embed = weights["pos_embed"];
    for (int i = 0; i < num_patches + 1; i++) {
      for (int j = 0; j < embed_dim; j++) {
        x(i, j) += pos_embed[i * embed_dim + j];
      }
    }

    for (int block_idx = 0; block_idx < depth; block_idx++) {
      string prefix = "blocks." + to_string(block_idx) + ".";

      Tensor x_norm = x;
      layer_norm(x_norm, weights[prefix + "norm1.weight"],
                 weights[prefix + "norm1.bias"]);

      Tensor attn_out = multihead_attention(x_norm, block_idx);
      for (int i = 0; i < x.size(); i++) {
        x.data[i] += attn_out.data[i];
      }

      x_norm = x;
      layer_norm(x_norm, weights[prefix + "norm2.weight"],
                 weights[prefix + "norm2.bias"]);

      Tensor mlp_out = mlp(x_norm, block_idx);
      for (int i = 0; i < x.size(); i++) {
        x.data[i] += mlp_out.data[i];
      }
    }

    layer_norm(x, weights["norm.weight"], weights["norm.bias"]);

    vector<float> cls_output(embed_dim);
    for (int j = 0; j < embed_dim; j++) {
      cls_output[j] = x(0, j);
    }

    Tensor head_weight =
        reshape_weight(weights["head.weight"], {num_classes, embed_dim});
    Tensor head_weight_T = transpose(head_weight);
    vector<float> &head_bias = weights["head.bias"];

    vector<float> logits(num_classes, 0.0f);
    for (int i = 0; i < num_classes; i++) {
      float sum = head_bias[i];
      for (int j = 0; j < embed_dim; j++) {
        sum += cls_output[j] * head_weight_T(j, i);
      }
      logits[i] = sum;
    }

    return logits;
  }

  int predict(const vector<float> &image) {
    auto logits = forward(image);
    /*cout << "\nLogits: ";
    for (int i = 0; i < logits.size(); i++) {
        cout << i << "=" << logits[i] << " ";
    }
    cout << endl;*/
    return max_element(logits.begin(), logits.end()) - logits.begin();
  }
};
