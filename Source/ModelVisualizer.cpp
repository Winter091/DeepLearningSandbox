#include "ModelVisualizer.hpp"

#include "bitmap_image.hpp"
#include <filesystem>
#include <cmath>


static void CreateBmp(const ModelLayer& layer, std::size_t index, const std::string& path)
{
    int width = std::round(std::sqrt(layer.PrevLayerSize));
    if (width * width < layer.PrevLayerSize) {
        ++width;
    }

    bitmap_image image(width, width);
    image.set_all_channels(0, 0, 0);

    for (int i = 0; i < layer.PrevLayerSize; i++) {
        int x = i % width;
        int y = i / width;

        float weight = layer.WeightAt(i, index);
        float coeff = std::abs(weight) / 2.0f;
        int r = 0, g = 0, b = 0;
        if (weight < 0) {
            r = coeff * 255;
        } else {
            g = coeff * 255;
        }

        image.set_pixel(x, y, r, g, b);
    }

    image.save_image(path);
}


void LayerToBmps(const ModelLayer& layer, const std::string& folder, int index)
{
    for (int i = 0; i < layer.Size; i++) {
        std::filesystem::path path(folder);
        path /= std::string("image_") + std::to_string(i) + std::string(".bmp");
        CreateBmp(layer, i, path.string());
    }
}
