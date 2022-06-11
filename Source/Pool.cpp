#include "Pool.hpp"

#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <random>


class FileDescriptor
{
public:
    FileDescriptor(const char* path, const char* mode) 
    { 
        m_file = fopen(path, mode);
        if (!m_file) {
            fprintf(stderr, "Unable to open file: %s\n", path);
            exit(EXIT_FAILURE);
        }
    }
    ~FileDescriptor() { fclose(m_file); }

    FILE* Get() { return m_file; }

private:
    FILE* m_file;
};


Pool::Pool(const char* imagesFile, const char* labelsFile)
{
    ParseImagesFile(imagesFile);
    ParseLabelsFile(labelsFile);
}


Pool Pool::TakeRandom(std::size_t amount) const
{
    std::vector<PoolElement> elements;
    std::unordered_set<uint32_t> takenElements;

    std::random_device os_seed;
    const uint32_t seed = os_seed();

    std::mt19937_64 generator(seed);
    std::uniform_int_distribution<uint32_t> random(0, std::numeric_limits<std::uint32_t>::max());

    while (elements.size() != amount) {
        uint32_t i = random(generator) % m_elements.size();
        if (takenElements.find(i) != takenElements.end()) {
            continue;
        }

        takenElements.insert(i);
        elements.push_back(m_elements[i]);
    }

    return Pool(std::move(elements));
}


void Pool::ParseLabelsFile(const char* labelsFile)
{
    FileDescriptor fileDescriptor(labelsFile, "rb");
    FILE* file = fileDescriptor.Get();

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

    std::vector<uint8_t> targets(poolSize);
    fread(&targets[0], targets.size(), 1, file);

    for (int i = 0; i < poolSize; i++) {
        m_elements[i].Target = targets[i];
    }
}


void Pool::ParseImagesFile(const char* imagesFile)
{
    FileDescriptor fileDescriptor(imagesFile, "rb");
    FILE* file = fileDescriptor.Get();

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

    std::vector<uint8_t> pixelsData(poolSize * pixelsPerImage);
    fread(&pixelsData[0], pixelsData.size(), 1, file);

    for (int i = 0; i < poolSize; i++) {
        auto& currElem = m_elements[i];
        currElem.Features.resize(pixelsPerImage);

        for (int j = 0; j < pixelsPerImage; j++) {
            uint8_t pixel = pixelsData[i * pixelsPerImage + j];
            currElem.Features[j] = (pixel / 255.0f);
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
