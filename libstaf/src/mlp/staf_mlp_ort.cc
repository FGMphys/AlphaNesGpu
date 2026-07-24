#include "staf_mlp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(STAF_WITH_ORT) && STAF_WITH_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace {

enum Act { ACT_LINEAR = 0, ACT_TANH = 1, ACT_RELU = 2 };

struct DenseLayer {
  int act;
  int in_f, out_f;
  std::vector<float> W_f, b_f; /* row-major [in, out] as Keras */
  std::vector<double> W_d, b_d;
};

struct TypeNet {
  int n_af = 0;
  int precision = 0; /* 0 float, 1 double */
  std::vector<float> mu_f;
  std::vector<double> mu_d;
  std::vector<DenseLayer> layers; /* native backend only */
#if defined(STAF_WITH_ORT) && STAF_WITH_ORT
  Ort::Session* session = nullptr;
  std::string onnx_path;
#endif
};

struct StafMlpImpl {
  StafMlpBackend backend;
  std::string model_dir;
  int precision = 0;
  int device_id = 0;
  std::vector<TypeNet> nets;
#if defined(STAF_WITH_ORT) && STAF_WITH_ORT
  Ort::Env* env = nullptr;
  Ort::SessionOptions* sopts = nullptr;
  bool cuda_ep = false;
#endif
};

template <typename T>
static void dense_forward(const DenseLayer& L, const T* x, T* y, int n_atoms,
                          bool is_double) {
  const int in_f = L.in_f, out_f = L.out_f;
  for (int a = 0; a < n_atoms; ++a) {
    const T* xa = x + a * in_f;
    T* ya = y + a * out_f;
    for (int j = 0; j < out_f; ++j) {
      T s = is_double ? (T)L.b_d[j] : (T)L.b_f[j];
      for (int i = 0; i < in_f; ++i) {
        T w = is_double ? (T)L.W_d[i * out_f + j] : (T)L.W_f[i * out_f + j];
        s += xa[i] * w;
      }
      if (L.act == ACT_TANH)
        s = std::tanh(s);
      else if (L.act == ACT_RELU)
        s = s > T(0) ? s : T(0);
      ya[j] = s;
    }
  }
}

template <typename T>
static void dense_backward(const DenseLayer& L, const T* x, const T* y,
                           const T* dy, T* dx, int n_atoms, bool is_double) {
  const int in_f = L.in_f, out_f = L.out_f;
  for (int a = 0; a < n_atoms; ++a) {
    const T* xa = x + a * in_f;
    const T* ya = y + a * out_f;
    const T* dya = dy + a * out_f;
    T* dxa = dx + a * in_f;
    for (int i = 0; i < in_f; ++i) dxa[i] = T(0);
    for (int j = 0; j < out_f; ++j) {
      T g = dya[j];
      if (L.act == ACT_TANH) {
        T t = ya[j];
        g *= (T(1) - t * t);
      } else if (L.act == ACT_RELU) {
        if (ya[j] <= T(0)) g = T(0);
      }
      for (int i = 0; i < in_f; ++i) {
        T w = is_double ? (T)L.W_d[i * out_f + j] : (T)L.W_f[i * out_f + j];
        dxa[i] += g * w;
      }
      (void)xa;
    }
  }
}

static int load_bin(TypeNet& net, const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "libstaf: cannot open %s\n", path);
    return -1;
  }
  char magic[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "STAFMLP1", 8) != 0) {
    fprintf(stderr, "libstaf: bad magic in %s\n", path);
    fclose(f);
    return -1;
  }
  int hdr[3];
  if (fread(hdr, sizeof(int), 3, f) != 3) {
    fclose(f);
    return -1;
  }
  net.precision = hdr[0];
  net.n_af = hdr[1];
  int n_layers = hdr[2];
  const bool is_d = net.precision == 1;
  if (is_d) {
    net.mu_d.resize(net.n_af);
    if (fread(net.mu_d.data(), sizeof(double), net.n_af, f) != (size_t)net.n_af) {
      fclose(f);
      return -1;
    }
  } else {
    net.mu_f.resize(net.n_af);
    if (fread(net.mu_f.data(), sizeof(float), net.n_af, f) != (size_t)net.n_af) {
      fclose(f);
      return -1;
    }
  }
  net.layers.resize(n_layers);
  for (int li = 0; li < n_layers; ++li) {
    int meta[3];
    if (fread(meta, sizeof(int), 3, f) != 3) {
      fclose(f);
      return -1;
    }
    DenseLayer& L = net.layers[li];
    L.act = meta[0];
    L.in_f = meta[1];
    L.out_f = meta[2];
    size_t nw = (size_t)L.in_f * (size_t)L.out_f;
    if (is_d) {
      L.W_d.resize(nw);
      L.b_d.resize(L.out_f);
      if (fread(L.W_d.data(), sizeof(double), nw, f) != nw ||
          fread(L.b_d.data(), sizeof(double), L.out_f, f) != (size_t)L.out_f) {
        fclose(f);
        return -1;
      }
    } else {
      L.W_f.resize(nw);
      L.b_f.resize(L.out_f);
      if (fread(L.W_f.data(), sizeof(float), nw, f) != nw ||
          fread(L.b_f.data(), sizeof(float), L.out_f, f) != (size_t)L.out_f) {
        fclose(f);
        return -1;
      }
    }
  }
  fclose(f);
  return 0;
}

/* n_af from type{k}_alpha_mu.dat (ORT path; no analytical weights needed). */
static int load_n_af_from_mu(TypeNet& net, const char* model_dir, int k,
                             int precision) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/type%d_alpha_mu.dat", model_dir, k);
  FILE* f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "libstaf: missing %s\n", path);
    return -1;
  }
  std::vector<double> vals;
  double v;
  while (fscanf(f, "%lf", &v) == 1) vals.push_back(v);
  fclose(f);
  if (vals.empty()) {
    fprintf(stderr, "libstaf: empty mu file %s\n", path);
    return -1;
  }
  net.n_af = (int)vals.size();
  net.precision = precision;
  if (precision == 1) {
    net.mu_d.assign(vals.begin(), vals.end());
  } else {
    net.mu_f.resize(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) net.mu_f[i] = (float)vals[i];
  }
  return 0;
}

template <typename T>
static int eval_type_native(TypeNet& net, const T* af, int n_atoms, T* energy,
                            T* dE_daf, T half_factor) {
  if (net.layers.empty()) {
    fprintf(stderr, "libstaf: native eval requires mlp_type*.bin weights\n");
    return -1;
  }
  const bool is_d = sizeof(T) == sizeof(double);
  const int n_af = net.n_af;
  std::vector<T> logdes((size_t)n_atoms * n_af);
  for (int a = 0; a < n_atoms; ++a) {
    for (int i = 0; i < n_af; ++i) {
      T v = af[a * n_af + i];
      T mu = is_d ? (T)net.mu_d[i] : (T)net.mu_f[i];
      logdes[a * n_af + i] = std::log(v + T(1e-3)) - mu;
    }
  }

  std::vector<std::vector<T>> acts;
  acts.push_back(logdes);
  for (size_t li = 0; li < net.layers.size(); ++li) {
    const DenseLayer& L = net.layers[li];
    std::vector<T> out((size_t)n_atoms * L.out_f);
    dense_forward<T>(L, acts.back().data(), out.data(), n_atoms, is_d);
    acts.push_back(std::move(out));
  }
  const std::vector<T>& atomic = acts.back();
  T e = T(0);
  for (int a = 0; a < n_atoms; ++a) e += atomic[a];
  *energy = half_factor * e;

  std::vector<T> delta(atomic.size(), T(1));
  for (int li = (int)net.layers.size() - 1; li >= 0; --li) {
    const DenseLayer& L = net.layers[li];
    std::vector<T> dx((size_t)n_atoms * L.in_f);
    dense_backward<T>(L, acts[li].data(), acts[li + 1].data(), delta.data(),
                      dx.data(), n_atoms, is_d);
    delta.swap(dx);
  }
  for (int a = 0; a < n_atoms; ++a) {
    for (int i = 0; i < n_af; ++i) {
      T v = af[a * n_af + i];
      dE_daf[a * n_af + i] = delta[a * n_af + i] / (v + T(1e-3));
    }
  }
  return 0;
}

#if defined(STAF_WITH_ORT) && STAF_WITH_ORT
static bool session_has_grad_outputs(Ort::Session* session) {
  if (!session) return false;
  size_t n_out = session->GetOutputCount();
  bool has_e = false, has_g = false;
  Ort::AllocatorWithDefaultOptions alloc;
  for (size_t i = 0; i < n_out; ++i) {
    auto name = session->GetOutputNameAllocated(i, alloc);
    if (std::strcmp(name.get(), "energy") == 0) has_e = true;
    if (std::strcmp(name.get(), "dE_daf") == 0) has_g = true;
  }
  return has_e && has_g;
}

template <typename T>
static int eval_type_ort(TypeNet& net, const T* af, int n_atoms, T* energy,
                         T* dE_daf) {
  if (!net.session) return -1;
  const int n_af = net.n_af;
  std::vector<int64_t> shape = {1, (int64_t)n_atoms, (int64_t)n_af};
  size_t n_elem = (size_t)n_atoms * (size_t)n_af;
  std::vector<T> af_bat(n_elem);
  std::memcpy(af_bat.data(), af, n_elem * sizeof(T));

  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<T>(
      mem, af_bat.data(), n_elem, shape.data(), shape.size());

  const char* in_names[] = {"af"};
  const char* out_names[] = {"energy", "dE_daf"};
  auto outs = net.session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1,
                               out_names, 2);
  const T* e_ptr = outs[0].GetTensorData<T>();
  *energy = e_ptr[0];
  const T* g_ptr = outs[1].GetTensorData<T>();
  auto ginfo = outs[1].GetTensorTypeAndShapeInfo();
  size_t g_count = ginfo.GetElementCount();
  if (g_count != n_elem) {
    fprintf(stderr, "libstaf ORT: dE_daf size %zu != %zu\n", g_count, n_elem);
    return -2;
  }
  std::memcpy(dE_daf, g_ptr, n_elem * sizeof(T));
  return 0;
}
#endif

}  // namespace

struct StafMlp {
  StafMlpImpl* impl;
};

extern "C" StafMlp* staf_mlp_create(StafMlpBackend backend, const char* model_dir,
                                    int precision, int device_id) {
  if (!model_dir) return NULL;
  auto* m = new StafMlp();
  m->impl = new StafMlpImpl();
  m->impl->backend = backend;
  m->impl->model_dir = model_dir;
  m->impl->precision = precision;
  m->impl->device_id = device_id;

  if (backend == STAF_MLP_ORT) {
#if !(defined(STAF_WITH_ORT) && STAF_WITH_ORT)
    fprintf(stderr, "libstaf: STAF_MLP_ORT requested but built without ORT\n");
    staf_mlp_destroy(m);
    return NULL;
#else
    for (int k = 0; k < 32; ++k) {
      char onnx[4096];
      snprintf(onnx, sizeof(onnx), "%s/model_type%d.onnx", model_dir, k);
      FILE* t = fopen(onnx, "rb");
      if (!t) break;
      fclose(t);
      TypeNet net;
      net.onnx_path = onnx;
      if (load_n_af_from_mu(net, model_dir, k, precision) != 0) {
        staf_mlp_destroy(m);
        return NULL;
      }
      m->impl->nets.push_back(std::move(net));
    }
    if (m->impl->nets.empty()) {
      fprintf(stderr,
              "libstaf ORT: no model_type*.onnx under %s "
              "(need export_mlp_grad_onnx.py)\n",
              model_dir);
      staf_mlp_destroy(m);
      return NULL;
    }

    m->impl->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "libstaf");
    m->impl->sopts = new Ort::SessionOptions();
    m->impl->sopts->SetIntraOpNumThreads(1);
    m->impl->cuda_ep = false;
    try {
      OrtCUDAProviderOptions cuda_opts{};
      cuda_opts.device_id = device_id;
      m->impl->sopts->AppendExecutionProvider_CUDA(cuda_opts);
      m->impl->cuda_ep = true;
      fprintf(stderr, "libstaf ORT: CUDA EP enabled (device %d)\n", device_id);
    } catch (const Ort::Exception& e) {
      fprintf(stderr, "libstaf ORT: CUDA EP failed (%s) — aborting (GPU required)\n",
              e.what());
      staf_mlp_destroy(m);
      return NULL;
    }
    try {
      for (auto& net : m->impl->nets) {
        net.session =
            new Ort::Session(*m->impl->env, net.onnx_path.c_str(), *m->impl->sopts);
        if (!session_has_grad_outputs(net.session)) {
          fprintf(stderr,
                  "libstaf ORT: %s missing energy/dE_daf outputs — "
                  "re-export with export_mlp_grad_onnx.py\n",
                  net.onnx_path.c_str());
          staf_mlp_destroy(m);
          return NULL;
        }
        fprintf(stderr, "libstaf ORT: loaded %s (n_af=%d)\n", net.onnx_path.c_str(),
                net.n_af);
      }
    } catch (const Ort::Exception& e) {
      fprintf(stderr, "libstaf ORT: session create failed: %s\n", e.what());
      staf_mlp_destroy(m);
      return NULL;
    }
    return m;
#endif
  }

  /* STAF_MLP_NATIVE (and TF_C placeholder): analytical Dense from .bin */
  for (int k = 0; k < 32; ++k) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/mlp_type%d.bin", model_dir, k);
    FILE* t = fopen(path, "rb");
    if (!t) break;
    fclose(t);
    TypeNet net;
    if (load_bin(net, path) != 0) {
      staf_mlp_destroy(m);
      return NULL;
    }
    m->impl->nets.push_back(std::move(net));
  }
  if (m->impl->nets.empty()) {
    fprintf(stderr, "libstaf: no mlp_type*.bin under %s\n", model_dir);
    staf_mlp_destroy(m);
    return NULL;
  }
  return m;
}

extern "C" int staf_mlp_eval(StafMlp* mlp, StafMlpEval* io) {
  if (!mlp || !mlp->impl || !io) return -1;
  StafMlpImpl* I = mlp->impl;
  const int n_type = (int)I->nets.size();
  if (io->n_atoms == nullptr || io->n_af == nullptr) return -1;

  if (I->precision == 0) {
    if (!io->af_f32 || !io->energy_f32 || !io->dE_daf_f32) return -1;
    float e_tot = 0.f;
    size_t af_off = 0, g_off = 0;
    for (int t = 0; t < n_type; ++t) {
      int na = io->n_atoms[t];
      int nf = io->n_af[t];
      if (nf != I->nets[t].n_af) return -3;
      float e = 0.f;
      int rc = -1;
      if (I->backend == STAF_MLP_ORT) {
#if defined(STAF_WITH_ORT) && STAF_WITH_ORT
        try {
          rc = eval_type_ort<float>(I->nets[t], io->af_f32 + af_off, na, &e,
                                    io->dE_daf_f32 + g_off);
        } catch (const Ort::Exception& ex) {
          fprintf(stderr, "libstaf ORT eval: %s\n", ex.what());
          return -5;
        }
#else
        return -6;
#endif
      } else {
        rc = eval_type_native<float>(I->nets[t], io->af_f32 + af_off, na, &e,
                                     io->dE_daf_f32 + g_off, 0.5f);
      }
      if (rc != 0) return -4;
      e_tot += e;
      af_off += (size_t)na * nf;
      g_off += (size_t)na * nf;
    }
    io->energy_f32[0] = e_tot;
    return 0;
  }

  if (!io->af_f64 || !io->energy_f64 || !io->dE_daf_f64) return -1;
  double e_tot = 0.0;
  size_t af_off = 0, g_off = 0;
  for (int t = 0; t < n_type; ++t) {
    int na = io->n_atoms[t];
    int nf = io->n_af[t];
    if (nf != I->nets[t].n_af) return -3;
    double e = 0.0;
    int rc = -1;
    if (I->backend == STAF_MLP_ORT) {
#if defined(STAF_WITH_ORT) && STAF_WITH_ORT
      try {
        rc = eval_type_ort<double>(I->nets[t], io->af_f64 + af_off, na, &e,
                                   io->dE_daf_f64 + g_off);
      } catch (const Ort::Exception& ex) {
        fprintf(stderr, "libstaf ORT eval: %s\n", ex.what());
        return -5;
      }
#else
      return -6;
#endif
    } else {
      rc = eval_type_native<double>(I->nets[t], io->af_f64 + af_off, na, &e,
                                    io->dE_daf_f64 + g_off, 0.5);
    }
    if (rc != 0) return -4;
    e_tot += e;
    af_off += (size_t)na * nf;
    g_off += (size_t)na * nf;
  }
  io->energy_f64[0] = e_tot;
  return 0;
}

extern "C" void staf_mlp_destroy(StafMlp* mlp) {
  if (!mlp) return;
  if (mlp->impl) {
#if defined(STAF_WITH_ORT) && STAF_WITH_ORT
    for (auto& net : mlp->impl->nets) {
      delete net.session;
      net.session = nullptr;
    }
    delete mlp->impl->sopts;
    delete mlp->impl->env;
#endif
    delete mlp->impl;
  }
  delete mlp;
}

extern "C" int staf_mlp_ntypes(const StafMlp* mlp) {
  if (!mlp || !mlp->impl) return 0;
  return (int)mlp->impl->nets.size();
}

extern "C" int staf_mlp_n_af(const StafMlp* mlp, int type) {
  if (!mlp || !mlp->impl || type < 0 ||
      type >= (int)mlp->impl->nets.size())
    return -1;
  return mlp->impl->nets[type].n_af;
}

extern "C" int staf_mlp_precision(const StafMlp* mlp) {
  if (!mlp || !mlp->impl) return -1;
  return mlp->impl->precision;
}
