#include "Pool.hpp"

#include <cstdio>
#include <cstdlib>


Pool::Pool(const char* imagesFile, const char* labelsFile)
{
    ParseImagesFile(imagesFile);
    ParseLabelsFile(labelsFile);
}


void Pool::ParseLabelsFile(const char* labelsFile)
{
    FILE* file = fopen(labelsFile, "rb");
    if (!file) {
        fprintf(stderr, "Unable to open labels file: %s\n", labelsFile);
        exit(EXIT_FAILURE);
    }

    fseek(file, 4, SEEK_SET);

    uint32_t poolSize;
    fread(&poolSize, 4, 1, file);
    poolSize = ReverseByteOrder(poolSize);

    if (m_size && poolSize != m_size) {
        fprintf(stderr, "Labels and images amounts don't match\n");
        exit(EXIT_FAILURE);
    }

    m_size = poolSize;
    m_elements.resize(poolSize);

    for (int i = 0; i < poolSize; i++) {
        uint8_t label;
        fread(&label, 1, 1, file);
        m_elements[i].Label = label;
    }
}


void Pool::ParseImagesFile(const char* imagesFile)
{
    FILE* file = fopen(imagesFile, "rb");
    if (!file) {
        fprintf(stderr, "Unable to open images file: %s\n", imagesFile);
        exit(EXIT_FAILURE);
    }

    fseek(file, 4, SEEK_SET);

    uint32_t poolSize;
    fread(&poolSize, 4, 1, file);
    poolSize = ReverseByteOrder(poolSize);

    if (m_size && poolSize != m_size) {
        fprintf(stderr, "Images and labels amounts don't match\n");
        exit(EXIT_FAILURE);
    }
    m_size = poolSize;
    m_elements.resize(poolSize);

    uint32_t rows, cols;
    fread(&rows, 4, 1, file);
    fread(&cols, 4, 1, file);

    rows = ReverseByteOrder(rows);
    cols = ReverseByteOrder(cols);

    uint32_t pixelsPerImage = rows * cols;

    for (auto& currElem : m_elements) {
        currElem.Pixels.resize(pixelsPerImage);

        for (int i = 0; i < pixelsPerImage; i++) {
            uint8_t pixel;
            fread(&pixel, 1, 1, file);
            currElem.Pixels[i] = (pixel / 255.0f);
        }
    }
}


uint32_t Pool::ReverseByteOrder(uint32_t num)
{
    uint32_t res = 0;
    res |= (num >> 24);
    res |= (((num >> 16) & 255) << 8);
    res |= (((num >> 8) & 255) << 16);
    res |= ((num & 255) << 24);

    return res;
}
