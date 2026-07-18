#include "usart_serial.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace
{
    UART_HandleTypeDef g_uart2_handle{};
    bool g_is_uart2_initialized = false;
}

extern "C" void HAL_UART_MspInit(UART_HandleTypeDef *huart)
{
    if (huart == nullptr || huart->Instance != USART2)
    {
        return;
    }

    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_USART2_CLK_ENABLE();

    GPIO_InitTypeDef gpio_init{};
    gpio_init.Pin = GPIO_PIN_2 | GPIO_PIN_3;
    gpio_init.Mode = GPIO_MODE_AF_PP;
    gpio_init.Pull = GPIO_PULLUP;
    gpio_init.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio_init.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(GPIOA, &gpio_init);
}

extern "C" void HAL_UART_MspDeInit(UART_HandleTypeDef *huart)
{
    if (huart == nullptr || huart->Instance != USART2)
    {
        return;
    }

    __HAL_RCC_USART2_CLK_DISABLE();
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_2 | GPIO_PIN_3);
}

void usart_serial_init(uint32_t baud_rate)
{
    g_uart2_handle.Instance = USART2;
    g_uart2_handle.Init.BaudRate = baud_rate;
    g_uart2_handle.Init.WordLength = UART_WORDLENGTH_8B;
    g_uart2_handle.Init.StopBits = UART_STOPBITS_1;
    g_uart2_handle.Init.Parity = UART_PARITY_NONE;
    g_uart2_handle.Init.Mode = UART_MODE_TX_RX;
    g_uart2_handle.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    g_uart2_handle.Init.OverSampling = UART_OVERSAMPLING_16;

    g_is_uart2_initialized = (HAL_UART_Init(&g_uart2_handle) == HAL_OK);
}

HAL_StatusTypeDef usart_serial_write(const uint8_t *data,
                                     std::size_t length,
                                     uint32_t timeout_ms)
{
    if (data == nullptr || length == 0U)
    {
        return HAL_OK;
    }

    if (!g_is_uart2_initialized)
    {
        usart_serial_init();
    }

    if (!g_is_uart2_initialized)
    {
        return HAL_ERROR;
    }

    constexpr std::size_t kMaxChunk = static_cast<std::size_t>(std::numeric_limits<uint16_t>::max());
    std::size_t offset = 0U;

    while (offset < length)
    {
        const std::size_t chunk_size = std::min(kMaxChunk, length - offset);
        HAL_StatusTypeDef status = HAL_UART_Transmit(&g_uart2_handle,
                                                     const_cast<uint8_t *>(&data[offset]),
                                                     static_cast<uint16_t>(chunk_size),
                                                     timeout_ms);
        if (status != HAL_OK)
        {
            return status;
        }

        offset += chunk_size;
    }

    return HAL_OK;
}

HAL_StatusTypeDef usart_serial_write_string(const char *text,
                                            uint32_t timeout_ms)
{
    if (text == nullptr)
    {
        return HAL_OK;
    }

    return usart_serial_write(reinterpret_cast<const uint8_t *>(text),
                              std::strlen(text),
                              timeout_ms);
}