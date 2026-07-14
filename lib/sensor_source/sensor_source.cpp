#include "sensor_source.h"
#include "sensor_source_data.h"

namespace
{
    constexpr std::size_t kSensorPhaseSamples =
        kSensorSampleRateHz * kSensorPhaseSeconds;

    std::size_t g_normal_offset = 0U;
    std::size_t g_ball_offset = 0U;
    std::size_t g_inner_offset = 0U;
    std::size_t g_outer_offset = 0U;
    std::size_t g_phase_index = 0U;
    std::size_t g_phase_samples_sent = 0U;

    struct SensorPhase
    {
        const int16_t *data;
        std::size_t *offset;
    };

    void copy_wrapped_frame(std::array<int16_t, kSensorRawSampleCount> &frame,
                            const int16_t *source,
                            std::size_t source_length,
                            std::size_t &offset)
    {
        if (offset + frame.size() > source_length)
        {
            offset = 0;
        }
        for (std::size_t idx = 0; idx < frame.size(); ++idx)
        {
            frame[idx] = source[offset];
            offset = (offset + 1U) % source_length;
        }
    }
} // namespace

void sensor_source_next_frame(std::array<int16_t, kSensorRawSampleCount> &frame)
{
    static SensorPhase phase_plan[] = {
        {kSensorNormalData, &g_normal_offset},
        {kSensorBallDefectData, &g_ball_offset},
        {kSensorNormalData, &g_normal_offset},
        {kSensorInnerDefectData, &g_inner_offset},
        {kSensorNormalData, &g_normal_offset},
        {kSensorOuterDefectData, &g_outer_offset},
    };

    SensorPhase &phase = phase_plan[g_phase_index];
    copy_wrapped_frame(frame,
                       phase.data,
                       kSensorStoredSamplesPerSignal,
                       *phase.offset);

    g_phase_samples_sent += frame.size();
    if (g_phase_samples_sent >= kSensorPhaseSamples)
    {
        g_phase_samples_sent = 0U;
        g_phase_index =
            (g_phase_index + 1U) % (sizeof(phase_plan) / sizeof(phase_plan[0]));
    }
}

std::size_t sensor_source_current_phase_index()
{
    return g_phase_index;
}
