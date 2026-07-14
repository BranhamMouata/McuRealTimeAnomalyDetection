#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

inline constexpr std::size_t kSensorRawSampleCount = 1024U;
inline constexpr std::size_t kSensorSampleRateHz = 48'000U;
inline constexpr std::size_t kSensorPhaseSeconds = 5U;
inline std::array<int16_t, kSensorRawSampleCount> sensor_buffer{};

void sensor_source_next_frame(std::array<int16_t, kSensorRawSampleCount> &frame);
std::size_t sensor_source_current_phase_index();
