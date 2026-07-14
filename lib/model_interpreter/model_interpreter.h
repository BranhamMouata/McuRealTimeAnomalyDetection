#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"

template <int N>
void fill_model_buffer(int8_t *i_inputs, const float *f_inputs,
                       int8_t qzero_point, float qscale)
{
  if (i_inputs == nullptr ||
      qscale <= 0.F)
  {
    return;
  }

  for (size_t idx = 0; idx < N; idx++)
  {
    const float quantized_f = std::nearbyint(f_inputs[idx] / qscale) +
                              static_cast<float>(qzero_point);
    constexpr int32_t kMinQ = static_cast<int32_t>(INT8_MIN);
    constexpr int32_t kMaxQ = static_cast<int32_t>(INT8_MAX);
    const int32_t quantized_i32 = std::clamp(static_cast<int32_t>(quantized_f),
                                             kMinQ,
                                             kMaxQ);
    i_inputs[idx] = static_cast<int8_t>(quantized_i32);
  }
}

void standardize_features(float *features);

float compute_reconstruction_error(
    const float *standardized_features,
    TfLiteTensor *output_tensor);

tflite::MicroInterpreter *get_model_interpreter();