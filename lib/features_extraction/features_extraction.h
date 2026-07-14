#pragma once
#include <cstdint>
#include <array>
#include <numeric>

struct Features
{
    float mean;
    float var;
    // root mean square
    float rms;
    // kurtosis
    float ku;
    // crest factor
    float cf;
    // impulse factor
    float impf;
    // peak to peak
    float p2p;
    // skewness
    float sw;
    // energy
    float en;
};

template <int DATA_SIZE>
Features extract_features(const std::array<float, DATA_SIZE> &raw_data)
{
    Features features{};
    if (raw_data.size() == 0)
    {
        return features;
    }
    // compute the mean
    features.mean = std::accumulate(raw_data.cbegin(), raw_data.cend(), 0.F) / raw_data.size();

    float sum_sq = 0.F;
    float sum_raw_sq = 0.F;
    float sum_cu = 0.F;
    float sum_qu = 0.F;
    float sum_abs = 0.F;
    float sum_sqrt = 0.F;
    float max_abs = 0.F;
    float min_value = raw_data.front();
    float max_value = raw_data.front();

    for (const float raw : raw_data)
    {
        const float centered = raw - features.mean;
        const float centered_sq = centered * centered;
        const float abs_value = std::abs(raw);

        sum_sq += centered_sq;
        sum_raw_sq += raw * raw;
        sum_cu += centered * centered_sq;
        sum_qu += centered_sq * centered_sq;
        sum_abs += abs_value;
        sum_sqrt += std::sqrt(abs_value);

        if (abs_value > max_abs)
        {
            max_abs = abs_value;
        }

        if (raw < min_value)
        {
            min_value = raw;
        }

        if (raw > max_value)
        {
            max_value = raw;
        }
    }

    // compute the variance
    features.var = sum_sq / raw_data.size();
    // compute root mean square
    features.rms = std::sqrt(sum_raw_sq / raw_data.size());
    // compute kurtosis
    features.ku = (features.var > 0.F) ? sum_qu / (features.var * features.var * raw_data.size()) : 0.F;
    // compute crest factor
    features.cf = (features.rms > 0.F) ? max_abs / features.rms : 0.F;
    // compute impulse factor
    const float mean_abs = sum_abs / raw_data.size();
    features.impf = (mean_abs > 0.F) ? max_abs / mean_abs : 0.F;
    // compute peak to peak
    features.p2p = max_value - min_value;
    // compute skewness
    features.sw = (features.var > 0.F) ? sum_cu / (std::sqrt(features.var) * features.var * raw_data.size()) : 0.F;
    // compute energy
    const float mean_sqrt = sum_sqrt / raw_data.size();
    features.en = mean_sqrt * mean_sqrt;
    return features;
}
