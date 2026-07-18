#include <stdint.h>
#include <numeric>
#include <array>

#include "stm32f4xx_hal.h"

#include "model_interpreter.h"
#include "scaler.h"
#include "error_threshold.h"
#include "sensor_source.h"
#include "features_extraction.h"
#include "sensor_source_data.h"
#ifdef INFER_FRAMES_RATE
#include "usart_serial.h"
#endif
#define LED_PIN GPIO_PIN_5
#define LED_GPIO_PORT GPIOA
#define LED_GPIO_CLK_ENABLE() __HAL_RCC_GPIOA_CLK_ENABLE()

// Raw floating point signal
static std::array<float, kSensorRawSampleCount> raw_signal{};

static void init_led_gpio()
{
  LED_GPIO_CLK_ENABLE();

  GPIO_InitTypeDef GPIO_InitStruct;

  GPIO_InitStruct.Pin = LED_PIN;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  HAL_GPIO_Init(LED_GPIO_PORT, &GPIO_InitStruct);
  HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN, GPIO_PIN_RESET);
}

int main(void)
{
  HAL_Init();
  init_led_gpio();
  // LiteRT model interpreter
  auto *interpreter = get_model_interpreter();
  auto *model_input_tensor = interpreter->input(0);
  auto *model_input_buffer = model_input_tensor->data.int8;
  auto *model_output_tensor = interpreter->output(0);
  const auto input_zero_point = model_input_tensor->params.zero_point;
  const auto input_scale = model_input_tensor->params.scale;
#ifdef INFER_FRAMES_RATE
  std::array<float, 10> infer_latence{};
  uint8_t infer_count = 0;
  float frame_rate = 0;
  usart_serial_init();
#endif

  while (1)
  {
    // Get sensor int16 signal
    sensor_source_next_frame(sensor_buffer);
#ifdef INFER_FRAMES_RATE
    uint32_t tickstart{};
    if (infer_count < infer_latence.size())
    {
      tickstart = HAL_GetTick();
    }
#endif
    // Convert int16 signal to floating point
    for (size_t idx = 0; idx < sensor_buffer.size(); ++idx)
    {
      raw_signal[idx] = static_cast<float>(sensor_buffer[idx]) / kSensorI16Scale;
    }
    // Extract features (mean, var, rms, kurtosis, crest factor, impulse factor, peak to peak, skewness, energy)
    auto features = extract_features<kSensorRawSampleCount>(raw_signal);
    // Standardize the features to fill in the model
    standardize_features(reinterpret_cast<float *>(&features));
    fill_model_buffer<kScalerFeatureCount>(model_input_buffer, reinterpret_cast<const float *>(&features),
                                           input_zero_point, input_scale);
    // Peform inference
    interpreter->Invoke();
    // Compute the reconstruction error
    const float reconstruction_error = compute_reconstruction_error(reinterpret_cast<const float *>(&features), model_output_tensor);
#ifdef INFER_FRAMES_RATE
    if (infer_count < infer_latence.size())
    {
      infer_latence[infer_count++] = HAL_GetTick() - tickstart;
    }
    else if (frame_rate == 0)
    {
      frame_rate = (infer_latence.size() * 1000) / std::accumulate(infer_latence.cbegin(), infer_latence.cend(), 0);
    }
    else
    {

      uint32_t frame_rate_int = static_cast<uint32_t>(frame_rate);
      const auto *text = (std::string("Frame rate: ") + std::to_string(frame_rate_int) +
                          std::string(" frames/sec") + std::string("\n"))
                             .c_str();
      usart_serial_write_string(text);
    }
#endif
    const GPIO_PinState led_state = (reconstruction_error >= kErrorThreshold) ? GPIO_PIN_SET : GPIO_PIN_RESET;
    HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN, led_state);
  }
}

extern "C"
{
  void SysTick_Handler(void)
  {
    HAL_IncTick();
  }

  void NMI_Handler(void)
  {
  }

  void HardFault_Handler(void)
  {
    while (1)
    {
    }
  }

  void MemManage_Handler(void)
  {
    while (1)
    {
    }
  }

  void BusFault_Handler(void)
  {
    while (1)
    {
    }
  }

  void UsageFault_Handler(void)
  {
    while (1)
    {
    }
  }

  void SVC_Handler(void)
  {
  }

  void DebugMon_Handler(void)
  {
  }

  void PendSV_Handler(void)
  {
  }
}
