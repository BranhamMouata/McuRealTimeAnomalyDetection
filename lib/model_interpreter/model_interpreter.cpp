#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "cwru_autoencoder_quant8.h"
#include "model_interpreter.h"

constexpr int kNumberOperators = 1;
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
