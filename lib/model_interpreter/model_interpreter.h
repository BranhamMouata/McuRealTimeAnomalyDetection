#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"

template <int N>
void fill_model_buffer(int8_t *i_inputs, const std::array<float, N> &f_inputs,
                       const float *standard_scaler_mean,
                       const float *standard_scaler_std,
                       int8_t qzero_point, float qscale)
{
  if (i_inputs == nullptr ||
      standard_scaler_mean == nullptr ||
      standard_scaler_std == nullptr ||
      qscale <= 0.F)
  {
    return;
  }

  for (size_t idx = 0; idx < f_inputs.size(); idx++)
  {
    const float std = standard_scaler_std[idx];
    const float standardized =
        (std != 0.F) ? ((f_inputs[idx] - standard_scaler_mean[idx]) / std) : 0.F;
    const float quantized_f = std::nearbyint(standardized / qscale) +
                              static_cast<float>(qzero_point);
    constexpr int32_t kMinQ = static_cast<int32_t>(INT8_MIN);
    constexpr int32_t kMaxQ = static_cast<int32_t>(INT8_MAX);
    const int32_t quantized_i32 = std::clamp(static_cast<int32_t>(quantized_f),
                                             kMinQ,
                                             kMaxQ);
    i_inputs[idx] = static_cast<int8_t>(quantized_i32);
  }
}

tflite::MicroInterpreter *get_model_interpreter();