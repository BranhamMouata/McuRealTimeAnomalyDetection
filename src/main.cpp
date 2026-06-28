#include <stdint.h>

#include "stm32f4xx_hal.h"

#include "model_interpreter.h"
#include "scaler.h"
#include "error_threshold.h"
#include <array>
#include <string>
#include "features_extraction.h"

#define LED_PIN GPIO_PIN_5
#define LED_GPIO_PORT GPIOA
#define LED_GPIO_CLK_ENABLE() __HAL_RCC_GPIOA_CLK_ENABLE()

constexpr uint8_t kSensorDataSize = kScalerFeatureCount;

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
  // get the interpreter
  auto *interpreter = get_model_interpreter();
  // Retrieve sensor data
  std::array<float, kSensorDataSize> sensor_data{};
  sensor_data.fill(1.F);

  // fill model input buffer
  auto *model_input_buffer = interpreter->input(0)->data.int8;
  const auto zero_point = interpreter->input(0)->params.zero_point;
  const auto scale = interpreter->input(0)->params.scale;
  fill_model_buffer<kSensorDataSize>(model_input_buffer, sensor_data, kScalerMean, kScalerStd, zero_point, scale);

  while (1)
  {
    HAL_GPIO_TogglePin(LED_GPIO_PORT, LED_PIN);
    // Perform inference
    interpreter->Invoke();
    HAL_Delay(1000);
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
