#pragma once

#include "tensorflow/lite/micro/micro_interpreter.h"

template <int N>
void fill_model_buffer(int8_t *i_inputs, const std::array<float, N> &f_inputs,
                       const float *standard_scaler_mean,
                       const float *standard_scaler_std,
                       int8_t qzero_point, float qscale)
{
  for (size_t idx = 0; idx < f_inputs.size(); idx++)
  {
    // standardize the input
    auto f_temp = (f_inputs[idx] - standard_scaler_mean[idx]) / standard_scaler_std[idx];
    // quantize the input
    constexpr auto min = static_cast<int32_t>(INT8_MIN);
    constexpr auto max = static_cast<int32_t>(INT8_MAX);
    *(i_inputs++) = static_cast<int8_t>(std::clamp(static_cast<int32_t>(static_cast<int32_t>(f_temp / qscale) + qzero_point),
                                                   min, max));
  }
}

tflite::MicroInterpreter *get_model_interpreter();