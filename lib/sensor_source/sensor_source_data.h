#pragma once

#include <cstddef>
#include <cstdint>

inline constexpr std::size_t kSensorStoredSamplesPerSignal = 48000U;
inline constexpr float kSensorI16Scale = 6.177433366e+03F;

extern const int16_t kSensorNormalData[kSensorStoredSamplesPerSignal];
extern const int16_t kSensorBallDefectData[kSensorStoredSamplesPerSignal];
extern const int16_t kSensorInnerDefectData[kSensorStoredSamplesPerSignal];
extern const int16_t kSensorOuterDefectData[kSensorStoredSamplesPerSignal];
