#include "ModelVisualizer.hpp"

#include "bitmap_image.hpp"
#include <filesystem>
#include <cmath>


void ModelVisualizer::DumpElemToBmp(
    const ModelLayer& layer, std::size_t layerElemIndex, const std::string& path)
{    
    int width = std::round(std::sqrt(layer.PrevLayerSize));
    if (width * width < layer.PrevLayerSize) {
        ++width;
    }

    bitmap_image image(width, width);
    image.set_all_channels(0, 0, 0);

    float maxAbsWeight = *std::max_element(layer.Weights.begin(), layer.Weights.end(), 
        [](float a, float b) {
            return std::abs(a) < std::abs(b);
        }
    );

    for (int i = 0; i < layer.PrevLayerSize; i++) {
        uint32_t x = i % width;
        uint32_t y = i / width;

        float weight = layer.WeightAt(i, layerElemIndex);
        float coeff = std::abs(weight / maxAbsWeight);

        uint8_t r = 0, g = 0, b = 0;
        if (weight < 0.0f) {
            r = coeff * 255.0f;
        } else {
            g = coeff * 255.0f;
        }

        image.set_pixel(x, y, r, g, b);
    }

    image.save_image(path);
}


void ModelVisualizer::DumpLayerToBmps(const ModelLayer& layer, const std::string& folder)
{
    for (int i = 0; i < layer.Size; i++) {
        std::filesystem::path path(folder);
        path /= std::string("weights_") + std::to_string(i) + std::string(".bmp");
        DumpElemToBmp(layer, i, path.string());
    }
}
