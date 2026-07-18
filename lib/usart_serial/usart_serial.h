#pragma once

#include <cstddef>
#include <cstdint>

#include "stm32f4xx_hal.h"

void usart_serial_init(uint32_t baud_rate = 115200U);

HAL_StatusTypeDef usart_serial_write(const uint8_t *data,
                                     std::size_t length,
                                     uint32_t timeout_ms = HAL_MAX_DELAY);

HAL_StatusTypeDef usart_serial_write_string(const char *text,
                                            uint32_t timeout_ms = HAL_MAX_DELAY);