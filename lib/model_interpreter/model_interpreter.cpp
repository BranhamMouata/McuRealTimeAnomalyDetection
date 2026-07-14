#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "cwru_autoencoder_quant8.h"
#include "model_interpreter.h"
#include "scaler.h"

constexpr int kNumberOperators = 1;
constexpr auto kFeaturesSize = kScalerFeatureCount;

using CwruOpResolver = tflite::MicroMutableOpResolver<kNumberOperators>;

static const TfLiteStatus RegisterOps(CwruOpResolver &op_resolver)
{
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  return kTfLiteOk;
}

static void interpreter_sanity_checks(tflite::MicroInterpreter &interpreter)
{
  TfLiteTensor *input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);
  TfLiteTensor *output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);
  TFLITE_CHECK_EQ(input->quantization.type, TfLiteQuantizationType::kTfLiteAffineQuantization);
  TFLITE_CHECK_EQ(output->quantization.type, TfLiteQuantizationType::kTfLiteAffineQuantization);
  TFLITE_CHECK_EQ(input->type, TfLiteType::kTfLiteInt8);
  TFLITE_CHECK_EQ(output->type, TfLiteType::kTfLiteInt8);
}

tflite::MicroInterpreter *get_model_interpreter()
{
  // Load the model and check the version of the model is compatible with the version of litert
  const tflite::Model *model =
      ::tflite::GetModel(cwru_autoencoder_quant8_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);
  // Register the operations
  static CwruOpResolver op_resolver;
  static bool is_op_resolver_initialized = false;
  if (!is_op_resolver_initialized)
  {
    TFLITE_CHECK_EQ(RegisterOps(op_resolver), kTfLiteOk);
    is_op_resolver_initialized = true;
  }
  // Allocate memory
  constexpr int kTensorArenaSize = 3000;
  static uint8_t tensor_arena[kTensorArenaSize];
  // Instantiate the interpreter and allocate tensors arena
  static tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                              kTensorArenaSize);
  TFLITE_CHECK_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  // Sanity checks
  interpreter_sanity_checks(interpreter);
  return &interpreter;
}

void standardize_features(
    float *features)
{
  for (size_t idx = 0; idx < kFeaturesSize; ++idx)
  {
    const float std = kScalerStd[idx];
    features[idx] = (std != 0.F)
                        ? ((features[idx] - kScalerMean[idx]) / kScalerStd[idx])
                        : 0.F;
  }
}

float compute_reconstruction_error(
    const float *standardized_features,
    TfLiteTensor *output_tensor)
{
  const auto output_zero_point = output_tensor->params.zero_point;
  const auto output_scale = output_tensor->params.scale;
  float squared_error_sum = 0.F;

  for (size_t idx = 0; idx < kFeaturesSize; ++idx)
  {
    const float reconstructed_feature =
        (static_cast<int32_t>(output_tensor->data.int8[idx]) - output_zero_point) * output_scale;
    const float error = standardized_features[idx] - reconstructed_feature;
    squared_error_sum += error * error;
  }

  return squared_error_sum / kFeaturesSize;
}